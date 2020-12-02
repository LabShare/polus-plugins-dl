from  bfio import  BioReader

import argparse, logging, sys
import numpy as np
from pathlib import Path

import zarr
import mxnet as mx
import models,utils

if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    ''' Argument parsing '''
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Cellpose parameters')
    
    # Input arguments
    parser.add_argument('--use_gpu', action='store_true', help='use gpu if mxnet with cuda installed', required=False)
    parser.add_argument('--diameter', dest='diameter', type=float,default=30.,help='Diameter', required=False)
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--pretrained_model', dest='pretrained_model', type=str,
                        help='Filename pattern used to separate data', required=False)
    parser.add_argument('--chan', required=False,
                        default=0, type=int, help='channel to segment; 0: GRAY, 1: RED, 2: GREEN, 3: BLUE')
    parser.add_argument('--chan2', required=False,
                        default=0, type=int,
                        help='nuclear channel (if cyto, optional); 0: NONE, 1: RED, 2: GREEN, 3: BLUE')
    # Output arguments
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    # Parse the arguments
    args = parser.parse_args()
    logger.info('diameter = {}'.format(args.diameter))
    inpDir = args.inpDir
    if (Path.is_dir(Path(args.inpDir).joinpath('images'))):
        # switch to images folder if present
        fpath = str(Path(args.inpDir).joinpath('images').absolute())
    logger.info('inpDir = {}'.format(inpDir))
    pretrained_model = args.pretrained_model
    logger.info('pretrained_model = {}'.format(pretrained_model))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    use_gpu = False

    if args.use_gpu:
        use_gpu = utils.use_gpu()
    if use_gpu:
        device = mx.gpu()
    else:
        device = mx.cpu()
    logger.info('Using %s'%(['CPU', 'GPU'][use_gpu]))

    if not (pretrained_model ):
            logger.info('Running the images on Cyto model')
            pretrained_model = 'cyto'

    model = models.Cellpose(device=device, model_type=pretrained_model)
    # Surround with try/finally for proper error catching
    try:
        # Start the javabridge with proper java logging
        logger.info('Initializing ...')
        log_config = Path(__file__).parent.joinpath("log4j.properties")
      #  jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(LOG4J)],class_path=JARS,run_headless=True)

        # Get all file names in inpDir image collection
        inpDir_files = [f.name for f in Path(inpDir).iterdir() if f.is_file() and "".join(f.suffixes) == '.ome.tif']
        channels = [args.chan, args.chan2]
        cstr0 = ['GRAY', 'RED', 'GREEN', 'BLUE']
        cstr1 = ['NONE', 'RED', 'GREEN', 'BLUE']
        logger.info('running cellpose on %d images using chan_to_seg %s and chan (opt) %s' %(len(inpDir_files), cstr0[channels[0]], cstr1[channels[1]]))
        if args.diameter == 0:

            diameter = None
            logger.info('Estimating diameter for each image')
        else:
            diameter = args.diameter
            logger.info(' Using diameter %0.2f for all images' % diameter)

        try:
            if Path(outDir).joinpath('location.zarr').exists():
                raise FileExistsError()

            root = zarr.group(store=str(Path(outDir).joinpath('location.zarr')))

            for f in inpDir_files:

                # Loop through files in inpDir image collection and process

                br = BioReader(str(Path(inpDir).joinpath(f).absolute()), max_workers=1)
                image = np.squeeze(br.read())

                if len(image.shape) >= 3:
                    if len(image.shape) == 4:
                        np.moveaxis(image, 2, 3)
                    prob_final = []
                    location_final = []
                    for i in range(image.shape[-1]):
                        location, prob = model.eval(image[:, :, i], channels=channels, diameter=diameter, image_name=f)
                        prob_final.append(prob.tolist())
                        location_final.append(location.tolist())

                    prob = np.asarray(prob_final)
                    location = np.asarray(location_final)

                elif len(image.shape) == 2:
                    location, prob = model.eval(image, channels=channels, diameter=diameter,
                                                image_name=f)

                cluster = root.create_group(f)
                init_cluster_1 = cluster.create_dataset('pixel_location', shape=location.shape, data=location)
                init_cluster_2 = cluster.create_dataset('probablity', shape=prob.shape, data=prob)
                cluster.attrs['metadata'] = str(br.metadata)
                del location, prob

        except FileExistsError:
            logger.info('Zarr file exists. Delete the existing file %r' % str((Path(outDir).joinpath('location.zarr'))))

    finally:
        # Close the javabridge regardless of successful completion
        logger.info('Closing the plugin')
      #  jutil.kill_vm()

        # Exit the program
        sys.exit()