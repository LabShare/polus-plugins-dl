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
from multiprocessing import cpu_count


######## Constants ########################################################################################
tiling_x = 508
tiling_y = 508
BG_VALUE = 1e20


def rescale(size,img,mode='uint8'):
    if mode == 'float32':
        #for floating point images:
        img = np.float32(img)
        img_PIL = PIL.Image.fromarray(img,mode='F')
    elif mode == 'uint8':
        img_PIL = PIL.Image.fromarray(img)
    else:
        raise(Exception('Invalid rescaling mode. Use uint8 or float32'))
            
    return np.array(img_PIL.resize(size,PIL.Image.BILINEAR))  #### PIL takes (width, height) as size


def normalize(img):
    ''' MIN/MAX-normalizes image to [0,1] range.'''
    ###normalize image
    img_min = np.min(img)
    img_max = np.max(img)
    img_centered = img - img_min
    img_range = img_max - img_min
    return np.true_divide(img_centered, img_range)


def createDataBlob(data, modelfile_path, img_pixelsize_x, img_pixelsize_y):
    ### prepare DATA Blob
    #get model resolution (element size) from modelfile
    modelfile_h5 = h5py.File(modelfile_path,'r')
    modelresolution_y = modelfile_h5['unet_param/element_size_um'][0]
    modelresolution_x = modelfile_h5['unet_param/element_size_um'][1]
    modelfile_h5.close()  
    #get input image absolute size
    abs_size_x = data.shape[1] * img_pixelsize_x        ### x --> WIDTH
    abs_size_y = data.shape[0] * img_pixelsize_y        ### y --> HEIGHT
    #get rescaled image size in pixel
    rescaled_size_px_x = int(np.round(abs_size_x / modelresolution_x))
    rescaled_size_px_y = int(np.round(abs_size_y / modelresolution_y))
    rescale_size = (rescaled_size_px_x,rescaled_size_px_y)

    #normalize image, then rescale
    normalized_img = normalize(data)
    rescaled_img = np.float32(rescale(rescale_size,normalized_img,mode='float32'))
    return rescaled_img


def getOutputTileShape(inputTileShape, modelfile_path):
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


def saveBlobs(rescaled_img, className, iofile_path, _weights, _labels, _samplepdf, foregroundbackgroundratio, borderWeightFactor, borderWeightSigmaPx, sigma1Px):
    
    iofile_h5 = h5py.File(iofile_path,mode='x')
    normalizationType = 1
    
    h5ready_img = rescaled_img[np.newaxis,np.newaxis,:,:]
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
    H = _data.shape[0]
    W = _data.shape[1]
    print(H, W)
    inShape = np.zeros([2])
    inShape[0] = tiling_x
    inShape[1] = tiling_y
    print(inShape)
    outShape = getOutputTileShape(inShape, modelfile_path)
    print(outShape)
    dataTile = np.zeros([int(inShape[0]), int(inShape[1])])
    labelsTile = np.zeros([int(outShape[0]), int(outShape[1])])
    weightsTile = np.zeros([int(outShape[0]), int(outShape[1])])
    print(dataTile.shape)
    print(weightsTile.shape)
    print(labelsTile.shape)
    tileOffset = np.zeros([int(inShape.size)])
    for d in range(tileOffset.size):
        tileOffset[d] = (inShape[d] - outShape[d]) / 2
    tiling = np.zeros([outShape.size])
    # nTiles = 1
    tiling[0] = int(math.ceil(float(H)/ float(outShape[0])))
    tiling[1] = int(math.ceil(float(W) / float(outShape[1])))
    print(tiling)
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

    min1Dist = np.zeros([H*W])
    min2Dist = np.zeros([H*W])
    extraWeights = np.zeros([H*W])
    va = 1.0 - foregroundbackgroundratio
    for i in range(H*W):
        min1Dist[i] = BG_VALUE
        min2Dist[i] = BG_VALUE

    for i in range(1, nComponents+1):
        instancelabels = np.zeros([H, W], dtype = np.uint8)
        idx = 0
        for x in range(H):
            for y in range(W):
                if instancelabelsdata[idx] == i:
                    instancelabels[x,y] = 0
                else:
                    instancelabels[x,y] = 255
                idx+=1
        d = np.zeros([H*W])
        dist = cv2.cv2.distanceTransform(instancelabels, cv2.cv2.DIST_L2, 3)
        idx = 0
        for x in range(H):
            for y in range(W):
                d[idx] = dist[x,y]
                idx+=1

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
    for ht in range(H):
        for wd in range(W):
            _weights[ht, wd] = weightsData[idx]
            _labels[ht, wd] = labelsData[idx]
            _samplepdf[ht, wd] = samplePdfData[idx]
            idx+=1

    return _weights, _labels, _samplepdf


def createlabelsandweightsfromrois(image, className, roiimage, foregroundbackgroundratio, borderWeightFactor, borderWeightSigmaPx, sigma1Px):
    # C = len(className) - 1
    W = image.shape[1]
    H = image.shape[0] 
    print(H, W)

    classlabelsdata = np.ones([H*W])
    instancelabelsdata = np.zeros([H*W])
    
    print(np.max(roiimage))
    roi_val = np.unique(roiimage)
    number = 1
    roiimage = roiimage.reshape(-1)
    for j in range(roi_val.shape[0]):
        if roi_val[j]>0:
            indices = np.where(roiimage == roi_val[j])
            for ind in indices:
                classlabelsdata[ind] = 2
                instancelabelsdata[ind] = number
            number+=1

    _weights, _labels, _samplepdf = addLabelsAndWeightsToBlobs(image, classlabelsdata, instancelabelsdata, number, foregroundbackgroundratio, borderWeightFactor, borderWeightSigmaPx, sigma1Px)
    
    return _weights, _labels, _samplepdf



def run_main(trainingImages, testingImages, pixelsize, foregroundbackgroundratio, borderWeightFactor, borderWeightSigmaPx, sigma1Px, outDir):

    img_pixelsize_x = int(pixelsize)
    img_pixelsize_y = int(pixelsize)
    foregroundbackgroundratio = float(foregroundbackgroundratio)
    borderWeightFactor = float(borderWeightFactor)
    borderWeightSigmaPx = float(borderWeightSigmaPx)
    sigma1Px = float(sigma1Px)

#    training_directory = "trainImages"
    modelfile_path = "2d_cell_net_v0-cytoplasm.modeldef.h5"

    ## Training Images
    className = []
    className.append("Background")
    className.append("cell")
    rootdir1 = Path(trainingImages)
    ind = 0
    try:     
        for PATH in rootdir1.glob('*img*'):
                mask_path = str(PATH).replace("img", "mask")
                mask_path = Path(mask_path)
                print(PATH, mask_path)
                tile_grid_size = 1
                tile_size = tile_grid_size * 1024
                with BioReader(PATH,backend='python',max_workers=cpu_count()) as br:
                    with BioReader(mask_path,backend='python',max_workers=cpu_count()) as br_label:
                        iofile_path = "_train_"+ str(ind) +".h5"
                        for t in range(br.T):

                            # Loop through channels
                            for c in range(br.C):

                                    # Loop through z-slices
                                    for z in range(br.Z):

                                        # Loop across the length of the image
                                        for y in range(0,br.Y,tile_size):
                                            y_max = min([br.Y,y+tile_size])

                                            # Loop across the depth of the image
                                            for x in range(0,br.X,tile_size):
                                                x_max = min([br.X,x+tile_size])
                                                img = np.squeeze(br[y:y_max,x:x_max,z:z+1,c,t])
                                                roiimage = np.squeeze(br_label[y:y_max,x:x_max,z:z+1,c,t])
                                                _data = createDataBlob(img, modelfile_path, img_pixelsize_x, img_pixelsize_y)
                                                _weights, _labels, _samplepdf = createlabelsandweightsfromrois(img, className, roiimage, foregroundbackgroundratio, borderWeightFactor, borderWeightSigmaPx, sigma1Px)
                                                saveBlobs(_data, className, iofile_path, _weights, _labels, _samplepdf, foregroundbackgroundratio, borderWeightFactor, borderWeightSigmaPx, sigma1Px)
                                                ind+=1

    finally:
        print("done")


    filename = 'trainfilelist.txt'
    file = open(filename, "w+")
    for i in range(ind):
        file.write("_train_"+str(i)+".h5"+"\n")
    file.close()

    rootdir1 = Path(testingImages)
    filename = 'validfilelist.txt'
    file = open(filename, "w+")
    total_valid = 0; ind = 0
    try:     
        for PATH in rootdir1.glob('*img*'):
                mask_path = str(PATH).replace("img", "mask")
                mask_path = Path(mask_path)
                print(PATH, mask_path)
                tile_grid_size = 1
                tile_size = tile_grid_size * 1024
                with BioReader(PATH,backend='python',max_workers=cpu_count()) as br:
                    with BioReader(mask_path,backend='python',max_workers=cpu_count()) as br_label:
                        valid_name = "_valid_"+str(ind) + "_"
                        for t in range(br.T):

                            # Loop through channels
                            for c in range(br.C):

                                    # Loop through z-slices
                                    for z in range(br.Z):

                                        # Loop across the length of the image
                                        for y in range(0,br.Y,tile_size):
                                            y_max = min([br.Y,y+tile_size])

                                            # Loop across the depth of the image
                                            for x in range(0,br.X,tile_size):
                                                x_max = min([br.X,x+tile_size])
                                                img = np.squeeze(br[y:y_max,x:x_max,z:z+1,c,t])
                                                roiimage = np.squeeze(br_label[y:y_max,x:x_max,z:z+1,c,t])
                                                _data = createDataBlob(img, modelfile_path, img_pixelsize_x, img_pixelsize_y)
                                                _weights, _labels, _samplepdf = createlabelsandweightsfromrois(img, className, roiimage, foregroundbackgroundratio, borderWeightFactor, borderWeightSigmaPx, sigma1Px)
                                                tiles = saveTiledBlobs(_data, _weights, _labels, modelfile_path, valid_name)
                                                for t in range(tiles):
                                                    file.write("_valid_" + str(ind)+ "_"+ str(t) + ".h5" + "\n")
                                                ind+=1
                                                total_valid += tiles

    finally:
        print("done")
        file.close()

    cf = CaffeSolver()
    cf.write("solver.prototxt", str(total_valid))

    solverPrototxtAbsolutePath = "solver.prototxt"
    weightfile_path = "caffemodels/2d_cell_net_v0.caffemodel.h5"
    run_unet_training(modelfile_path, weightfile_path,solverPrototxtAbsolutePath, outDir)