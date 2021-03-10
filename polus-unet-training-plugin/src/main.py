import argparse, logging, subprocess, time, multiprocessing, sys
from pathlib import Path
from bfio_finetuning import run_main

if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    ''' Argument parsing '''
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='WIPP plugin to train UNet model from UFreiburg')
    
    # Input arguments
    parser.add_argument('--borderWeightFactor', dest='borderWeightFactor', type=str, default="50.0",
                        help='lambda separation', required=False)
    parser.add_argument('--borderWeightSigmaPx', dest='borderWeightSigmaPx', type=str, default="6.0",
                        help='Sigma for balancing weight function.', required=False)
    parser.add_argument('--foregroundbackgroundgratio', dest='foregroundbackgroundgratio', type=str, default="0.1",
                        help='Foreground/Background ratio', required=False)
    parser.add_argument('--pixelsize', dest='pixelsize', type=str,
                        help='Input image pixel size', required=True)
    parser.add_argument('--sigma1Px', dest='sigma1Px', type=str, default="10.0",
                        help='Sigma for instance segmentation.', required=False)
    parser.add_argument('--testingImages', dest='testingImages', type=str,
                        help='Input testing image collection to be processed by this plugin', required=True)
    parser.add_argument('--trainingImages', dest='trainingImages', type=str,
                        help='Input training image collection to be processed by this plugin', required=True)
    # Output arguments
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    borderWeightFactor = args.borderWeightFactor
    logger.info('borderWeightFactor = {}'.format(borderWeightFactor))
    borderWeightSigmaPx = args.borderWeightSigmaPx
    logger.info('borderWeightSigmaPx = {}'.format(borderWeightSigmaPx))
    foregroundbackgroundgratio = args.foregroundbackgroundgratio
    logger.info('foregroundbackgroundgratio = {}'.format(foregroundbackgroundgratio))
    pixelsize = args.pixelsize
    logger.info('pixelsize = {}'.format(pixelsize))
    sigma1Px = args.sigma1Px
    logger.info('sigma1Px = {}'.format(sigma1Px))
    testingImages = args.testingImages
    if (Path.is_dir(Path(args.testingImages).joinpath('images'))):
        # switch to images folder if present
        fpath = str(Path(args.testingImages).joinpath('images').absolute())
    logger.info('testingImages = {}'.format(testingImages))
    trainingImages = args.trainingImages
    if (Path.is_dir(Path(args.trainingImages).joinpath('images'))):
        # switch to images folder if present
        fpath = str(Path(args.trainingImages).joinpath('images').absolute())
    logger.info('trainingImages = {}'.format(trainingImages))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    run_main(trainingImages, testingImages, pixelsize, foregroundbackgroundgratio, borderWeightFactor, borderWeightSigmaPx, sigma1Px, outDir)
    