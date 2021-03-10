import h5py
import numpy as np
import PIL
import subprocess
import os
from PIL import Image
import sys
from bfio import BioReader, BioWriter, LOG4J, JARS
from pathlib import Path
from multiprocessing import cpu_count
import json

def rescale(size,img,mode='uint8'):
    
    if mode == 'float32':
        #for floating point images:
        img = np.float32(img)
        img_PIL = PIL.Image.fromarray(img,mode='F')
    elif mode == 'uint8':
        #otherwise:
        img_PIL = PIL.Image.fromarray(img)
    else:
        raise(Exception('Invalid rescaling mode. Use uint8 or float32'))
          
    return np.array(img_PIL.resize(size,PIL.Image.BILINEAR))


def normalize(img):

    ###normalize image
    img_min = np.min(img)
    img_max = np.max(img)
    img_centered = img - img_min
    img_range = img_max - img_min
    return np.true_divide(img_centered, img_range)


def unet_segmentation(input_img,img_pixelsize_x,img_pixelsize_y,
                          modelfile_path,weightfile_path,iofile_path,
                          tiling_x=4,tiling_y=4,gpu_flag='',
                          cleanup=True):

    #fix parameters
    n_inputchannels=1
    n_iterations=0
    
    
    ## prepare image rescaling
    np.set_printoptions(threshold=sys.maxsize)

    #get model resolution (element size) from modelfile
    modelfile_h5 = h5py.File(modelfile_path,'r')
    modelresolution_y = modelfile_h5['unet_param/element_size_um'][0]
    modelresolution_x = modelfile_h5['unet_param/element_size_um'][1]
    modelfile_h5.close()       
    #get input image absolute size
    abs_size_x = input_img.shape[1] * img_pixelsize_x
    abs_size_y = input_img.shape[0] * img_pixelsize_y
    #get rescaled image size in pixel
    rescaled_size_px_x = int(np.round(abs_size_x / modelresolution_x))
    rescaled_size_px_y = int(np.round(abs_size_y / modelresolution_y))
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

    # ### run caffe_unet commands

    # #assemble sanity check command
    command_sanitycheck = []
    command_sanitycheck.append("caffe_unet")
    command_sanitycheck.append("check_model_and_weights_h5")
    command_sanitycheck.append("-model")
    command_sanitycheck.append(modelfile_path)
    command_sanitycheck.append("-weights")
    command_sanitycheck.append(weightfile_path)
    command_sanitycheck.append("-n_channels")
    command_sanitycheck.append(str(n_inputchannels))
    if gpu_flag:
        command_sanitycheck.append("-gpu")
        command_sanitycheck.append(gpu_flag)
     #runs command and puts console output to stdout
    sanitycheck_proc = subprocess.run(command_sanitycheck,stdout=subprocess.PIPE)
    # #aborts if process failed
    sanitycheck_proc.check_returncode()
    #assemble prediction command
    command_predict = []
    command_predict.append("caffe_unet")
    command_predict.append("tiled_predict")
    command_predict.append("-infileH5")
    command_predict.append(iofile_path)
    command_predict.append("-outfileH5")
    command_predict.append(iofile_path)
    command_predict.append("-model")
    command_predict.append(modelfile_path)
    command_predict.append("-weights")
    command_predict.append(weightfile_path)
    command_predict.append("-iterations")
    command_predict.append(str(n_iterations))
    command_predict.append("-n_tiles")
    command_predict.append(str(tiling_x)+'x'+str(tiling_y))
    command_predict.append("-gpu")
    command_predict.append(gpu_flag)
    if gpu_flag:
        command_predict.append("-gpu")
        command_predict.append(gpu_flag)
    #run command 
    try:
        output = subprocess.check_output(command_predict, stderr=subprocess.STDOUT).decode()
        print(output)
    except subprocess.CalledProcessError as e:
        print(e.output.decode()) # print out the stdout messages up to the exception
        print(e)

    # load results from io file and return
    output_h5 = h5py.File(iofile_path)
    score = output_h5['score'][:]
    output_h5.close()
    # #get segmentation mask by taking channel argmax
    segmentation_mask = np.squeeze(np.argmax(score, axis=1))
    return segmentation_mask


def run_segmentation(inpDir, pixelsize, weights, filename, outDir):

    img_pixelsize_x = int(pixelsize)                 
    img_pixelsize_y = int(pixelsize)
    modelfile_path = "2d_cell_net_v0-cytoplasm.modeldef.h5"
    weightfile_path = str(Path(weights)/filename)
    iofile_path = "output.h5"
    out_path = Path(outDir)
    rootdir1 = Path(inpDir)
    """ Convert the tif to tiled tiff """
    i = 0
    try:
        for PATH in rootdir1.glob('**/*'):
                tile_grid_size = 1
                tile_size = tile_grid_size * 1024

                # Set up the BioReader
                with BioReader(PATH,backend='python',max_workers=cpu_count()) as br:
 
                    # Loop through timepoints
                    for t in range(br.T):

                        # Loop through channels
                        for c in range(br.C):

                            with BioWriter(out_path.joinpath(f"out{i}.ome.tif"),metadata = br.metadata, backend='python') as bw:

                                 # Loop through z-slices
                                for z in range(br.Z):

                                    # Loop across the length of the image
                                    for y in range(0,br.Y,tile_size):
                                        y_max = min([br.Y,y+tile_size])

                                        # Loop across the depth of the image
                                        for x in range(0,br.X,tile_size):
                                            x_max = min([br.X,x+tile_size])

                                            input_img = np.squeeze(br[y:y_max,x:x_max,z:z+1,c,t])
                                            img = unet_segmentation(input_img,img_pixelsize_x, img_pixelsize_y,modelfile_path,weightfile_path,iofile_path)
                                            bw[y:y_max, x:x_max, z:z+1, 0, 0] = img.astype(br.dtype)
                                            os.remove("output.h5")
                                            print("output.h5 removed.")
                            i+=1

    finally:
        print("Bfio exit.")
