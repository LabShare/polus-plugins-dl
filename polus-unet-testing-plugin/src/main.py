# from bfio.bfio import BioReader, BioWriter
# import bioformats
# import javabridge as jutil
import argparse, logging, subprocess, time, multiprocessing, sys
# import numpy as np
from pathlib import Path
from unet_test import run_segmentation

if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    ''' Argument parsing '''
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='WIPP plugin to test UNet model from UFreiburg')
    
    # Input arguments
    parser.add_argument('--filename', dest='filename', type=str,
                        help='Weights file for testing.', required=True)
    parser.add_argument('--input_directory', dest='input_directory', type=str,
                        help='Input image collection to be processed by this plugin.', required=True)
    parser.add_argument('--pixelsize', dest='pixelsize', type=str,
                        help='Input image pixel size.', required=True)
    parser.add_argument('--weights', dest='weights', type=str,
                        help='Weights path for testing.', required=True)
    # Output arguments
    parser.add_argument('--output_directory', dest='output_directory', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    filename = args.filename
    logger.info('filename = {}'.format(filename))
    input_directory = args.input_directory
    if (Path.is_dir(Path(args.input_directory).joinpath('images'))):
        # switch to images folder if present
        fpath = str(Path(args.input_directory).joinpath('images').absolute())
    logger.info('input_directory = {}'.format(input_directory))
    pixelsize = args.pixelsize
    logger.info('pixelsize = {}'.format(pixelsize))
    weights = args.weights
    logger.info('weights = {}'.format(weights))
    output_directory = args.output_directory
    logger.info('output_directory = {}'.format(output_directory))
    run_segmentation(input_directory, pixelsize, weights, filename, output_directory)