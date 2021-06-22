import argparse, logging, subprocess, time, multiprocessing, sys
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
    parser.add_argument('--weightsfilename', dest='weightsfilename', type=str,
                        help='Weights file name for testing.', required=True)
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin.', required=True)
    parser.add_argument('--filePattern', dest='filePattern', type=str,
                        help='Filename pattern to filter data.', required=True)
    parser.add_argument('--pixelsize', dest='pixelsize', type=str,
                        help='Input image pixel size in micrometers.', required=False)
    parser.add_argument('--weights', dest='weights', type=str,
                        help='Weights file path for testing.', required=True)
    # Output arguments
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    weightsfilename = args.weightsfilename
    logger.info('weightsfilename = {}'.format(weightsfilename))
    inpDir = args.inpDir
    if (Path.is_dir(Path(args.inpDir).joinpath('images'))):
        # switch to images folder if present
        fpath = str(Path(args.inpDir).joinpath('images').absolute())
    logger.info('inpDir = {}'.format(inpDir))
    filePattern = args.filePattern
    logger.info('filePattern = {}'.format(filePattern))
    pixelsize = args.pixelsize
    logger.info('pixelsize = {}'.format(pixelsize))
    weights = args.weights
    logger.info('weights = {}'.format(weights))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    run_segmentation(inpDir,filePattern, pixelsize, weights, weightsfilename, outDir)
    logger.info('Inference completed.')