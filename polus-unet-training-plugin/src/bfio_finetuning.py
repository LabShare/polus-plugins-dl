import h5py
import numpy as np
import PIL
import subprocess
import os, sys
from PIL import Image, ImageSequence
import cv2
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import math
from pathlib import Path
from solver_example import CaffeSolver
from train import run_unet_training

from bfio import BioReader, BioWriter, LOG4J, JARS
from pathlib import Path
import pathlib
from multiprocessing import cpu_count
import logging
import filepattern


######## Constants ########
tiling_x = 508
tiling_y = 508
BG_VALUE = 1e20
logger = logging.getLogger("preprocessing")
logger.setLevel(logging.INFO)

def normalize(img):
    """Normalizes the original image.

    Normalizes the input image based on min/max normalization.

    Args:
        img: input image.

    Returns:
        Normalized image.
    """
    
    logger.info("Normalizing image ...")
    img_min = np.min(img)
    img_max = np.max(img)
    img_centered = img - img_min
    img_range = img_max - img_min

    return np.true_divide(img_centered, img_range)


def getOutputTileShape(inputTileShape, modelfile_path):
    """Get output tile shape for validation image tiles.

    Args:
        inputTileShape: input image tile shape
        modelfile_path: model file

    Returns:
        Output tile shape.
    """

    modelfile_h5 = h5py.File(modelfile_path,'r')
    res = np.zeros([2])
    padInput = np.zeros([2])
    padOutput = np.zeros([2])
    padInput[0] = modelfile_h5["/unet_param/padInput"][0]
    padInput[1] = modelfile_h5["/unet_param/padInput"][0]
    padOutput[0] = modelfile_h5["/unet_param/padOutput"][0]
    padOutput[1] = modelfile_h5["/unet_param/padOutput"][0]
    modelfile_h5.close()
    for d in range(2):
        res[d] = inputTileShape[d] - (padInput[d] - padOutput[d])

    return res


def saveBlobs(_data, className, iofile_path, _weights, _labels, _samplepdf, foregroundbackgroundratio, borderWeightFactor, borderWeightSigmaPx, sigma1Px):
    """Save training blobs.

    Args:
        _data: Normalized training image
        className: Class names
        iofile_path: training H5 file name to save blobs
        _weights: training weights
        _labels: training labels
        _samplepdf: training probability distribution 
        foregroundbackgroundratio: foreground to background ratio
        borderWeightFactor: lambda separation
        borderWeightSigmaPx: Sigma for balancing weight function
        sigma1Px: Sigma for instance segmentation
    """

    logger.info("Saving training image blobs ...")
    iofile_h5 = h5py.File(iofile_path,mode='x')
    normalizationType = 1
    
    h5ready_img = _data[np.newaxis,np.newaxis,:,:]
    _weights = _weights[np.newaxis,np.newaxis,:,:]
    _labels = _labels[np.newaxis,np.newaxis,:,:]
    _samplepdf = _samplepdf[np.newaxis,np.newaxis,:,:]

    iofile_h5.create_dataset('data',data=h5ready_img)
    iofile_h5.create_dataset('weights',data=_weights)
    iofile_h5.create_dataset('labels',data=_labels)
    iofile_h5.create_dataset('weights2',data=_samplepdf)
    grp1 = iofile_h5.create_group("conversionParameters")
    grp1.attrs["foregroundBackgroundRatio"] = foregroundbackgroundratio
    grp1.attrs["sigma1_um"] = sigma1Px
    grp1.attrs["borderWeightFactor"] = borderWeightFactor
    grp1.attrs["borderWeightSigmaPx"] = borderWeightSigmaPx
    grp1.attrs["normalizationType"] = normalizationType
    grp1.attrs["classNames"] = className
    iofile_h5.close()


def saveTiledBlobs(_data, _weights, _labels, modelfile_path, valid_name):
    """Save validation blobs.

    Args:
        _data: Normalized training image
        _weights: training weights
        _labels: training labels
        modelfile_path: model file
        valid_name: file name to save validation blob

    Returns:
        Number of validation tiles
    """

    logger.info("Saving validation tiled blobs ...")
    H = _data.shape[0]
    W = _data.shape[1]
    inShape = np.zeros([2])
    inShape[0] = tiling_x
    inShape[1] = tiling_y
    logger.info("inShape = {}".format(inShape))
    outShape = getOutputTileShape(inShape, modelfile_path)
    logger.info("outShape = {}".format(outShape))
    dataTile = np.zeros([int(inShape[0]), int(inShape[1])])
    labelsTile = np.zeros([int(outShape[0]), int(outShape[1])])
    weightsTile = np.zeros([int(outShape[0]), int(outShape[1])])
    tileOffset = np.zeros([int(inShape.size)])
    for d in range(tileOffset.size):
        tileOffset[d] = (inShape[d] - outShape[d]) / 2
    tiling = np.zeros([outShape.size])
    tiling[0] = int(math.ceil(float(H)/ float(outShape[0])))
    tiling[1] = int(math.ceil(float(W) / float(outShape[1])))
    logger.info("tiling = {}".format(tiling))
    tileIdx = 0
    for yIdx in range(int(tiling[0])):
        for xIdx in range(int(tiling[1])):
            for y in range(int(inShape[0])):
                yR = yIdx * outShape[0] - tileOffset[0] + y
                if (yR < 0): yR = -yR
                n = int(yR / (H - 1))
                yR = (yR - n * (H - 1)) if (n % 2 == 0)  else ((n + 1) * (H - 1) - yR)
                for x in range(int(inShape[1])):
                    xR = xIdx * outShape[1] - tileOffset[1] + x
                    if (xR < 0): xR = -xR
                    n = int(xR / (W - 1))
                    xR = (xR - n * (W - 1)) if (n % 2 == 0) else ((n + 1) * (W - 1) - xR)
                    dataTile[y, x] = _data[int(yR), int(xR)]

            for y in range(int(outShape[0])):
                yR = yIdx * outShape[0] + y
                n = int(yR / (H - 1))
                yR = (yR - n * (H - 1)) if (n % 2 == 0) else ((n + 1) * (H - 1) - yR)
                for x in range(int(outShape[1])):
                    xR = xIdx * outShape[1] + x
                    n = int(xR / (W - 1))
                    xR = (xR - n * (W - 1)) if (n % 2 == 0) else ((n + 1) * (W - 1) - xR)
                    labelsTile[y,x] = _labels[int(yR), int(xR)]
                    if (xR == xIdx * outShape[1] + x and yR == yIdx * outShape[0] + y):
                        weightsTile[y, x] = _weights[int(yR), int(xR)]
                    else:
                        weightsTile[y, x] = float(0.0)

            iofile_path = valid_name + str(tileIdx) + ".h5"
            iofile_h5 = h5py.File(iofile_path,mode='x')
            _datatile = dataTile[np.newaxis,np.newaxis,:,:]
            _weightsTile = weightsTile[np.newaxis,np.newaxis,:,:]
            _labelsTile = labelsTile[np.newaxis,np.newaxis,:,:]
            iofile_h5.create_dataset('data3',data=_datatile)
            iofile_h5.create_dataset('weights',data=_weightsTile)
            iofile_h5.create_dataset('labels',data=_labelsTile)
            iofile_h5.close()
            tileIdx+=1
    return tileIdx


def addLabelsAndWeightsToBlobs(image, classlabelsdata, instancelabelsdata, nComponents, foregroundbackgroundratio, borderWeightFactor, borderWeightSigmaPx, sigma1Px):
    """Create labels, weights and probability distribution.

    Args:
        image: input image 
        classlabelsdata: class labels array
        instancelabelsdata: instance labels array
        nComponents: total number of instances
        foregroundbackgroundratio: foreground to background ratio
        borderWeightFactor: lambda separation
        borderWeightSigmaPx: Sigma for balancing weight function
        sigma1Px: Sigma for instance segmentation

    Returns:
        labels, weights, probability distribution
    """

    logger.info("Creating labels and weights ...")
    W = image.shape[1]
    H = image.shape[0] 
    D = 1
    z = 0
    dy = [ -1,  0,  1, -1 ]
    dx = [ -1, -1, -1,  0 ]
    dz = [ 0,  0,  0,  0 ]  
    idx = -1
    labelsData = np.zeros([H*W])
    weightsData = np.zeros([H*W])
    samplePdfData = np.zeros([H*W])
    for i in range(H*W):
        labelsData[i] = float(0.0)
        weightsData[i] = float(-1.0)
        samplePdfData[i] = float(foregroundbackgroundratio)

    for x in range(H):
        for y in range(W):
            idx+=1
            classLabel = classlabelsdata[idx]
            if (classLabel == 0): 
                weightsData[idx] = float(0.0)
                samplePdfData[idx] = float(0.0)
            elif (classLabel == 1):
                pass
            else:
                instanceLabel = instancelabelsdata[idx]
                nbIdx = 0
                for nbIdx in range(len(dx)):
                    if (z + dz[nbIdx] < 0 or y + dy[nbIdx] < 0 or y + dy[nbIdx] >= W or x + dx[nbIdx] < 0 or x + dx[nbIdx] >= H):
                        pass
                    else:
                        nbInst = instancelabelsdata[(x + dx[nbIdx]) * W + y + dy[nbIdx]]
                        if (nbInst > 0 and nbInst != instanceLabel): break
                nbIdx+=1
                if (nbIdx == len(dx)):
                    labelsData[idx] = classLabel - 1
                    weightsData[idx] = float(1.0)
                    samplePdfData[idx] = float(1.0)

    min1Dist = [BG_VALUE for i in range(H*W)]
    min2Dist = [BG_VALUE for i in range(H*W)]
    
    extraWeights = np.zeros([H*W])
    va = 1.0 - foregroundbackgroundratio

    logger.info("nComponents = {}".format(nComponents))

    for i in range(1, nComponents+1):
        instancelabels = np.zeros([H, W], dtype = np.uint8)
        instancelabels = np.reshape(np.where(instancelabelsdata == i, 0, 255), (H, W))
        d = np.zeros([H*W])
        dist = cv2.cv2.distanceTransform(np.uint8(instancelabels), cv2.cv2.DIST_L2, 3)
        d = dist.flatten()
        for j in range(H*W):
            min1dist = min1Dist[j]
            min2dist = min(min2Dist[j], float(d[j]))
            min1Dist[j] = min(min1dist, min2dist)
            min2Dist[j] = max(min1dist, min2dist)

    for z in range(D):
        for i in range(H*W):
            if (weightsData[i] >= float(0.0)): continue
            d1 = min1Dist[z * W * H + i]
            d2 = min2Dist[z * W * H + i]
            wa = math.exp(-(d1 * d1) / (2 * sigma1Px * sigma1Px))
            we = math.exp(-(d1 + d2) * (d1 + d2) /(2 * borderWeightSigmaPx * borderWeightSigmaPx))
            extraWeights[z * H * W + i] += borderWeightFactor * we + va * wa
    
    for z in range(D):
      for i in range(H*W):
        if (weightsData[i] >= float(0.0)): continue
        weightsData[i] = float(foregroundbackgroundratio) + extraWeights[z * H * W + i]
    
    idx = 0
    _weights = np.ones([H,W])
    _samplepdf = np.ones([H,W])
    _labels = np.ones([H,W])
    _weights = np.reshape(weightsData, (H,W))
    _labels = np.reshape(labelsData, (H,W))
    _samplepdf = np.reshape(samplePdfData, (H,W))

    return _weights, _labels, _samplepdf


def createlabelsandweightsfromrois(image, roiimage):
    """Create class labels and instance labels.

    Args:
        image: input image
        roiimage: input mask image

    Returns:
        classlabelsdata, instancelabelsdata, total_instances
    """

    logger.info("Creating class and instance labels ...")
    W = image.shape[1]
    H = image.shape[0]
    logger.info("H, W = {},{}".format(H, W))
    classlabelsdata = np.ones([H*W])
    instancelabelsdata = np.zeros([H*W])
    
    roi_val = np.unique(roiimage)
    total_instances = 1
    roiimage = roiimage.reshape(-1)
    for j in range(roi_val.shape[0]):
        if roi_val[j]>0:
            indices = np.where(roiimage == roi_val[j])
            for ind in indices:
                classlabelsdata[ind] = 2
                instancelabelsdata[ind] = total_instances
            total_instances+=1

    return classlabelsdata, instancelabelsdata, total_instances


def run_main(trainingImages, testingImages, trainingLabels, testingLabels, pixelsize, iterations, foregroundbackgroundratio, borderWeightFactor, borderWeightSigmaPx, sigma1Px, outDir):
    """Main function to read input images, prepare data blobs and run training.

    Args:
        trainingImages: input training images
        testingImages: input validation images
        trainingLabels: input training labels
        testingLabels: input validation labels
        pixelsize: image pixel size
        iterations: number of training iterations
        foregroundbackgroundratio: foreground to background ratio
        borderWeightFactor: lambda separation
        borderWeightSigmaPx: Sigma for balancing weight function
        sigma1Px: Sigma for instance segmentation
        outDir: output directory to save results and trained model
    """
    img_pixelsize_x = float(pixelsize)
    img_pixelsize_y = float(pixelsize)
    foregroundbackgroundratio = float(foregroundbackgroundratio)
    borderWeightFactor = float(borderWeightFactor)
    borderWeightSigmaPx = float(borderWeightSigmaPx)
    sigma1Px = float(sigma1Px)
    modelfile_path = "2d_cell_net_v0-cytoplasm.modeldef.h5"

    f1 = h5py.File(modelfile_path, 'r+')
    data = f1['unet_param/element_size_um']       
    data[...] = [img_pixelsize_x, img_pixelsize_y]
    f1.close()   
    f1 = h5py.File(modelfile_path, 'r')
    logger.info('Model Element Size {}'.format(f1['unet_param/element_size_um'][...]))
    f1.close()
    
    ## Training Images
    className = []
    className.append("Background")
    className.append("cell")
    rootdir1 = Path(trainingImages)
    ind = 0
    try:     
        filepath = pathlib.Path(__file__).parent
        filepath = filepath.joinpath(rootdir1)
        for file in filepath.iterdir():
            img_path = Path(file)
            mask_path = Path(trainingLabels+"/"+file.name)
            logger.info('Processing training image {}'.format(img_path))
            logger.info('Processing training mask {}'.format(mask_path))
            tile_grid_size = 1
            tile_size = tile_grid_size * 1024
            with BioReader(img_path,backend='python',max_workers=cpu_count()) as br:
                with BioReader(mask_path,backend='python',max_workers=cpu_count()) as br_label:
                    # Loop through z-slices
                    for z in range(br.Z):

                        # Loop across the length of the image
                        for y in range(0,br.Y,tile_size):
                            y_max = min([br.Y,y+tile_size])

                            # Loop across the depth of the image
                            for x in range(0,br.X,tile_size):
                                iofile_path = "_train_"+ str(ind) +".h5"
                                x_max = min([br.X,x+tile_size])
                                img = np.squeeze(br[y:y_max,x:x_max,z:z+1,0,0])
                                roiimage = np.squeeze(br_label[y:y_max,x:x_max,z:z+1,0,0])
                                _data = normalize(img)
                                classlabelsdata, instancelabelsdata, total_instances = createlabelsandweightsfromrois(img, roiimage)
                                _weights, _labels, _samplepdf = addLabelsAndWeightsToBlobs(img, classlabelsdata, instancelabelsdata, total_instances, foregroundbackgroundratio, borderWeightFactor, borderWeightSigmaPx, sigma1Px)
                                saveBlobs(_data, className, iofile_path, _weights, _labels, _samplepdf, foregroundbackgroundratio, borderWeightFactor, borderWeightSigmaPx, sigma1Px)
                                ind+=1

    finally:
        logger.info("training blobs created.")


    filename = 'trainfilelist.txt'
    trainingfile = open(filename, "w+")
    for i in range(ind):
        trainingfile.write("_train_"+str(i)+".h5"+"\n")
    trainingfile.close()

    ## Validation Images
    rootdir2 = Path(testingImages)
    filename = 'validfilelist.txt'
    validationfile = open(filename, "w+")
    total_valid = 0; ind = 0
    try:     
        filepath = pathlib.Path(__file__).parent
        filepath = filepath.joinpath(rootdir2)
        # i = 0
        for file in filepath.iterdir():
            img_path = Path(file)
            mask_path = Path(testingLabels+"/"+file.name)
            logger.info('Processing validation image {}'.format(img_path))
            logger.info('Processing validation mask {}'.format(mask_path))
            tile_grid_size = 1
            tile_size = tile_grid_size * 1024
            with BioReader(img_path,backend='python',max_workers=cpu_count()) as br:
                with BioReader(mask_path,backend='python',max_workers=cpu_count()) as br_label:
                    # Loop through z-slices
                    for z in range(br.Z):

                        # Loop across the length of the image
                        for y in range(0,br.Y,tile_size):
                            y_max = min([br.Y,y+tile_size])

                            # Loop across the depth of the image
                            for x in range(0,br.X,tile_size):
                                valid_name = "_valid_"+str(ind) + "_"
                                x_max = min([br.X,x+tile_size])
                                img = np.squeeze(br[y:y_max,x:x_max,z:z+1,0,0])
                                roiimage = np.squeeze(br_label[y:y_max,x:x_max,z:z+1,0,0])
                                _data = normalize(img)
                                classlabelsdata, instancelabelsdata, total_instances = createlabelsandweightsfromrois(img, roiimage)
                                _weights, _labels, _samplepdf = addLabelsAndWeightsToBlobs(img, classlabelsdata, instancelabelsdata, total_instances, foregroundbackgroundratio, borderWeightFactor, borderWeightSigmaPx, sigma1Px)
                                tiles = saveTiledBlobs(_data, _weights, _labels, modelfile_path, valid_name)
                                for t in range(tiles):
                                    validationfile.write("_valid_" + str(ind)+ "_"+ str(t) + ".h5" + "\n")
                                ind+=1
                                total_valid += tiles

    finally:
        logger.info("validation blobs created.")
        validationfile.close()

    ## Create Solver file
    logger.info("creating solver file ...")
    cf = CaffeSolver()
    cf.write("solver.prototxt", str(total_valid), iterations)
    solverPrototxtAbsolutePath = "solver.prototxt"

    ## Run Unet Training
    logger.info("training started ...")
    weightfile_path = "caffemodels/2d_cell_net_v0.caffemodel.h5"
    run_unet_training(modelfile_path, weightfile_path,solverPrototxtAbsolutePath, outDir)
    os.system("cp snapshot_iter_"+iterations+".caffemodel.h5 "+outDir)
    logger.info("training completed.")