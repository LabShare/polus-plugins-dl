from  bfio import  BioReader

import argparse, logging, sys,random
import numpy as np
from pathlib import Path
import os
import zarr
import models,utils

def read (inpDir,flow_path,image_names):
    flow_list=[]
    image_all=[]
    root = zarr.open(str(Path(flow_path).joinpath('flow.zarr')), mode='r')
    for f in image_names:
        br = BioReader(str(Path(inpDir).joinpath(f).absolute()))
        mask_name= str(str(f).split('.',1)[0]+'_mask.'+str(f).split('.',1)[1])
        if mask_name not in root.keys():
            print('%s not present in zarr file'%mask_name)
        image_all.append(np.squeeze(br.read()))
        flow_list.append(root[mask_name]['flow'])
     #   print(root[mask_name]['flow'][0,...].shape)
    return image_all,flow_list



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
    parser.add_argument('--unet', required=False,
                        default=0, type=int, help='run standard unet instead of cellpose flow output')
    parser.add_argument('--diameter', dest='diameter', type=float,default=30.,help='Diameter', required=False)
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--pretrained_model', dest='pretrained_model', type=str,
                        help='Filename pattern used to separate data', required=False)
    parser.add_argument('--cpmodel_path', dest='cpmodel_path', type=str,
                        help='Filename pattern used to separate data', required=False)
    parser.add_argument('--flow_path',help='Flow path should be a zarr file', type=str, required=True)
    parser.add_argument('--train_size', action='store_true', help='train size network at end of training')
    parser.add_argument('--test_dir', required=False,
                        default=[], type=str, help='folder containing test data (optional)')
    parser.add_argument('--learning_rate', required=False,
                        default=0.2, type=float, help='learning rate')
    parser.add_argument('--n_epochs', required=False,
                        default=500, type=int, help='number of epochs')
    parser.add_argument('--batch_size', required=False,
                        default=8, type=int, help='batch size')
    parser.add_argument('--residual_on', required=False,
                        default=1, type=int, help='use residual connections')
    parser.add_argument('--style_on', required=False,
                        default=1, type=int, help='use style vector')
    parser.add_argument('--concatenation', required=False,
                        default=0, type=int,
                        help='concatenate downsampled layers with upsampled layers (off by default which means they are added)')
    parser.add_argument('--train_fraction', required=False,
                        default=0.8, type=float, help='test train split')
    parser.add_argument('--nclasses', required=False,
                        default=3, type=int, help='if running unet, choose 2 or 3, otherwise not used')
    # Output arguments
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    # Parse the arguments
    args = parser.parse_args()
    diameter=args.diameter
    logger.info('diameter = {}'.format(diameter))
    inpDir = args.inpDir
    if (Path.is_dir(Path(args.inpDir).joinpath('images'))):
        # switch to images folder if present
        fpath = str(Path(args.inpDir).joinpath('images').absolute())
    logger.info('inpDir = {}'.format(inpDir))
    pretrained_model = args.pretrained_model
    logger.info('pretrained_model = {}'.format(pretrained_model))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    flow_path=args.flow_path
    cpmodel_path= args.cpmodel_path
    train_fraction=args.train_fraction
    model_dir = Path.home().joinpath('.cellpose', 'models')

    if pretrained_model == 'cyto' or pretrained_model == 'nuclei':
        torch_str = 'torch'
        cpmodel_path = os.fspath(model_dir.joinpath('%s%s_0' % (pretrained_model, torch_str)))
        if pretrained_model == 'cyto':
            szmean = 30.
        else:
            szmean = 17.
    else:
        cpmodel_path = os.fspath(pretrained_model)
        szmean = 30

    if  not Path(cpmodel_path).exists():
        cpmodel_path = False
        print('>>>> training from scratch')
        if diameter == 0:
            rescale = False
            print('>>>> median diameter set to 0 => no rescaling during training')
        else:
            rescale = True
            szmean = diameter
    else:
        rescale = True
        diameter = szmean
        print('>>>> pretrained model %s is being used' % cpmodel_path)
        args.residual_on = 1
        args.style_on = 1
        args.concatenation = 0


    model = models.CellposeModel(pretrained_model=cpmodel_path,diam_mean=szmean,residual_on=args.residual_on,style_on=args.style_on,concatenation=args.concatenation)
    # Surround with try/finally for proper error catching
    try:
        # Start the javabridge with proper java logging
        logger.info('Initializing ...')

        # Get all file names in inpDir image collection
       # channels = [args.chan, args.chan2]
        channels =[0,0]

        image_names = [f.name for f in Path(inpDir).iterdir() if f.is_file() and "".join(f.suffixes) == '.tif'  ]
        random.shuffle(image_names)
        idx = int(train_fraction * len(image_names))
        train_img_names = image_names[0:idx]
        test_img_names = image_names[idx:]
        logger.info('running cellpose on %d images ' %(len(image_names)))
        diameter = args.diameter
        logger.info(' Using diameter %0.2f for all images' % diameter)


        try:
            if not Path(flow_path).joinpath('flow.zarr').exists():
                raise FileExistsError()

            # train data
            train_images,train_labels = read(inpDir,flow_path,train_img_names)
            # test data
            test_images,test_labels  = read(inpDir,flow_path,test_img_names)

            cpmodel_path = model.train(train_images, train_labels, train_files=train_img_names,
                                           test_data=test_images, test_labels=test_labels, test_files=test_img_names,
                                           learning_rate=args.learning_rate, channels=channels,
                                           save_path=outDir, rescale=rescale,
                                           n_epochs=args.n_epochs,
                                           batch_size=args.batch_size)
            print('>>>> model trained and saved to %s' % cpmodel_path)

        except FileExistsError:
            logger.info('Zarr file does not exist. File not found in path  %r' % str((Path(inpDir).joinpath('flow.zarr'))))

    finally:
        # Close the javabridge regardless of successful completion
        logger.info('Closing the plugin')
      #  jutil.kill_vm()

        # Exit the program
        sys.exit()