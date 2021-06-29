import h5py
import numpy as np
import PIL
import os
from PIL import Image
import sys
from bfio import BioReader, BioWriter
from pathlib import Path
from multiprocessing import cpu_count
import json
import filepattern

UNITS = {'m':  10**6,
         'cm': 10**4,
         'mm': 10**3,
         'µm': 1,
         'nm': 10**-3,
         'pm': 10**-6}

modelfile_path = "2d_cell_net_v0-cytoplasm.modeldef.h5"
#get model resolution (element size) from modelfile
modelfile_h5 = h5py.File(modelfile_path,'r')
modelresolution_y = modelfile_h5['unet_param/element_size_um'][0]
modelresolution_x = modelfile_h5['unet_param/element_size_um'][1]
modelfile_h5.close()    

def rescale(size,img,mode='uint8'):
    """Rescales the normalized image.

    Rescales image if image pixel size is different from model resolution.

    Args:
        size: rescale factor in pixel.
        img: normalized image.

    Returns:
        Rescaled image.
    """

    if mode == 'float32':
        #for floating point images:
        img = np.float32(img)
        img_PIL = PIL.Image.fromarray(img,mode='F')
    elif mode == 'uint8':
        #otherwise:
        img_PIL = PIL.Image.fromarray(img)
    else:
        raise(Exception('Invalid rescaling mode. Use uint8 or float32'))
          
    return np.array(img_PIL.resize(size,PIL.Image.NEAREST))


def normalize(img):
    """Normalizes the original image.

    Normalizes the input image based on min/max normalization.

    Args:
        img: normalized image.

    Returns:
        Normalized image.
    """

    img_min = np.min(img)
    img_max = np.max(img)
    img_centered = img - img_min
    img_range = img_max - img_min
    return np.true_divide(img_centered, img_range)


def unet_segmentation(input_img,img_pixelsize_x,img_pixelsize_y,weightfile_path,iofile_path,
                          cleanup=True):
    """Run unet segmentation.

    Preprocess image and run unet segmentation using caffe binary. 
    If rescale factor is greater than 10, the code will throw an error.

    Args:
        input_img: input image to be segmented.
        img_pixelsize_x: input image pixel size in x.
        img_pixelsize_y: input image pixel size in y.
        weightfile_path: weight file path.
        iofile_path: input/output file to save the preprocessed image and final segmentation score.

    Returns:
        Segmentation Mask.
    """

    #fixed parameters
    n_iterations=0
    tiling_x=4
    tiling_y=4

    ## prepare image rescaling
    np.set_printoptions(threshold=sys.maxsize)
    #get input image absolute size
    abs_size_x = input_img.shape[1]
    abs_size_y = input_img.shape[0]
    rescale_factor_x = img_pixelsize_x / modelresolution_x
    rescale_factor_y = img_pixelsize_y / modelresolution_y

    if rescale_factor_x < 10 and rescale_factor_y < 10:
        rescaled_size_px_x = int(np.round(abs_size_x*rescale_factor_x))
        rescaled_size_px_y = int(np.round(abs_size_y*rescale_factor_y))
    else:
        raise Exception('rescale factor greater than 10.')

    if rescale_factor_x > 1 or rescale_factor_y > 1:
        tiling_x=3*rescale_factor_x
        tiling_y=3*rescale_factor_y

    rescale_size = (rescaled_size_px_x,rescaled_size_px_y)
    ### preprocess image and store in IO file
    #normalize image, then rescale
    normalized_img = normalize(input_img)
    rescaled_img = np.float32(rescale(rescale_size,normalized_img,mode='float32'))
    #prepending singleton dimensions to get the desired blob structure
    h5ready_img = np.expand_dims(rescaled_img, axis=(0,1))
    iofile_h5 = h5py.File(iofile_path,mode='x')
    iofile_h5.create_dataset('data',data=h5ready_img)
    iofile_h5.close()

    ### run caffe_unet commands
    cmd = "caffe_unet tiled_predict -infileH5 "+iofile_path+" -outfileH5 "+iofile_path+" -model "+modelfile_path+\
          " -weights "+weightfile_path+" -iterations "+str(n_iterations)+" -n_tiles "+str(tiling_x)+"x"+str(tiling_y)
    os.system(cmd)
    # load results from io file and return
    output_h5 = h5py.File(iofile_path)
    score = output_h5['score'][:]
    output_h5.close()
    #get segmentation mask by taking channel argmax
    segmentation_mask = np.squeeze(np.argmax(score, axis=1))
    return segmentation_mask


def run_segmentation(inpDir, filePattern, pixelsize, weights, weightsfilename, outDir):
    """Save segmentation mask using Biowriter.

    Read input image, call segmentation function and save output mask using Biowriter. 
    If pixel size is not provided, it will be read from image metadata.
    If no metadata is available, model resolution will be used as pixel size.

    Args:
        inpDir: input image directory.
        filePattern: input file name pattern to filter data.
        pixelsize: image pixel size.
        weights: weights file folder.
        weightsfilename: weights file name.
        outDir: output directory to save masks.
    """
    img_pixelsize_x = img_pixelsize_y = None if pixelsize is None else float(pixelsize)
    weightfile_path = str(Path(weights)/weightsfilename)
    iofile_path = "output.h5"
    out_path = Path(outDir)
    rootdir = Path(inpDir)   
    fp = filepattern.FilePattern(rootdir,filePattern)
    

    """ Convert the tif to tiled tiff """
    for fP in fp():
        for PATH in fP:
            print(PATH.get("file"))
            tile_grid_size = 1
            tile_size = tile_grid_size * 2048

            # Set up the BioReader
            with BioReader(PATH.get("file")) as br:

                with BioWriter(out_path.joinpath(PATH.get("file").name),metadata = br.metadata, backend='python') as bw:
                    
                    # Set pixel size based on metadata from each image if it exists
                    if img_pixelsize_x is None and br.ps_x[0] is not None:
                        img_pixelsize_x = br.ps_x[0] * UNITS[br.ps_x[1]]
                        img_pixelsize_y = br.ps_y[0] * UNITS[br.ps_y[1]]
                    elif img_pixelsize_x is None:
                        # Set to the model resolution so no resizing occurs
                        img_pixelsize_x = modelresolution_x
                        img_pixelsize_y = modelresolution_y

                        # Loop through z-slices
                    for z in range(br.Z):
                        # Loop across the length of the image
                        for y in range(0,br.Y,tile_size):
                            y_max = min([br.Y,y+tile_size])
                            # Loop across the depth of the image
                            for x in range(0,br.X,tile_size):
                                x_max = min([br.X,x+tile_size])
                                input_img = np.squeeze(br[y:y_max,x:x_max,z:z+1,0,0])
                                img = unet_segmentation(input_img,img_pixelsize_x, img_pixelsize_y,weightfile_path,iofile_path)
                                rescale_size_output = (input_img.shape[1], input_img.shape[0])
                                img = rescale(rescale_size_output,img,mode='float32')
                                bw.dtype = np.uint8
                                bw[y:y_max, x:x_max, z:z+1, 0, 0] = img.astype(np.uint8)
                                os.remove("output.h5")
