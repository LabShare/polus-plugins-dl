from bfio import BioReader, BioWriter
import argparse, logging, subprocess, time, multiprocessing, sys, os, typing
import traceback
import numpy as np
from pathlib import Path
import torch 
import torchvision
from preprocess import LocalNorm
import filepattern

TILE_SIZE = 1024
TILE_OVERLAP = 256

def pad_image(img, multiple=128):
    """Pad input image to make it a certain zie

    Args:
        img (array): input image
        multiple (int, optional): make size a multiple of

    Returns:
        array: padded image
        tuple: tuple consisting of pad dimensions
    """

    pad_x = multiple - (img.shape[0]%multiple) if img.shape[0]%multiple!=0 else 0
    pad_y = multiple - (img.shape[1]%multiple) if img.shape[1]%multiple!=0 else 0
    padded_img = np.pad(img, [(0,pad_x),(0,pad_y)], mode='reflect') 
    return padded_img, (pad_x,pad_y)

def postprocess(out_img, segmentationType):
    """postprocessing the output image

    Args:
        out_img (array): output image
        segmentationType (str): segmentation type

    Returns:
        [array]: output image after poat processing
    """
    # reshape output
    c,x,y = out_img.shape
    out_img = out_img.reshape((x,y,1,c))

    # binary segmentation used sigmoid activation
    if segmentationType == 'Binary':
        out_img[out_img>=0.5] = 255
        out_img[out_img<0.5] = 0
    return out_img

    
if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    ''' Argument parsing '''
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Segmentation models inference plugin')
    
    # Input arguments
    parser.add_argument('--pattern', dest='pattern', type=str,
                        help='Filename pattern used to separate data', required=True)
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--modelPath', dest='modelPath', type=str,
                        help='pretrained model to use', required=True)
    parser.add_argument('--segmentationType', dest='segmentationType', type=str,
                        help='Segmentation Type', required=True)

    # Output arguments
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    pattern = args.pattern
    logger.info('pattern = {}'.format(pattern))
    modelPath = args.modelPath
    logger.info('modelPath = {}'.format(modelPath))
    inpDir = args.inpDir
    if (Path.is_dir(Path(args.inpDir).joinpath('images'))):
        # switch to images folder if present
        fpath = str(Path(args.inpDir).joinpath('images').absolute())
    logger.info('inpDir = {}'.format(inpDir))
    segmentationType = args.segmentationType
    logger.info('segmentationType = {}'.format(segmentationType))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))

    # pretrained model
    model_path = os.path.join(modelPath, 'out_model.pth')

    # initialize preprocessing
    preprocess = torchvision.transforms.Compose([
           torchvision.transforms.ToTensor(),
           LocalNorm() 
           ])

    # change based on segmentation type
    out_dtype = np.uint8 if segmentationType=='Binary' else np.float32
    backend = 'python' if segmentationType=='Binary' else 'zarr'
    classes = 1 if segmentationType=='Binary' else 3

    # Surround with try/finally for proper error catching
    try:
        # device
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        logger.info('using device: {}'.format(dev))

        # load model
        logger.info('loading pretrained model')
        model = torch.load(model_path).to(dev)
        model.eval()

        fp = filepattern.FilePattern(file_path=inpDir, pattern=pattern)
         
        # Loop through files in inpDir image collection and process
        for f in fp():
            file_name = f[0]['file']
            logger.info('Processing image: {}'.format(file_name.name))
            out_file_name = file_name if segmentationType=='Binary' else Path(str(file_name).replace('.ome.tif', '_flow.ome.zarr'))

            with BioReader(file_name) as br, \
                 BioWriter(Path(outDir).joinpath(Path(out_file_name).name), metadata=br.metadata, backend=backend) as bw:
                bw.dtype = out_dtype
                bw.C = classes
                if classes == 3:
                    bw.cnames = ['cell_probability', 'x', 'y']

                # iterate over tiles
                for x in range(0,br.X,TILE_SIZE):
                    x_min = max([0,x-TILE_OVERLAP])
                    x_max = min([br.X,x+TILE_SIZE+TILE_OVERLAP])
                    x_left_trim = x-x_min
                    x_right_trim = x_max - min([br.X,x+TILE_SIZE])

                    for y in range(0,br.Y,TILE_SIZE):
                        y_min = max([0,y-TILE_OVERLAP])
                        y_max = min([br.Y,y+TILE_SIZE+TILE_OVERLAP])
                        y_left_trim = y-y_min
                        y_right_trim = y_max - min([br.Y,y+TILE_SIZE])

                        # read image
                        img = br[y_min:y_max,x_min:x_max,0,0,0]

                        # pad image if required to make dimensions a multiple of 128
                        pad_dims = None
                        if not (img.shape[0]%128==0 and img.shape[1]%128==0):
                            img, pad_dims = pad_image(img)
                        
                        # preprocess image
                        img = img.astype(np.float32)
                        img = preprocess(img).unsqueeze(0).to(dev)

                        with torch.no_grad():
                            out = model(img).cpu().numpy()
                        
                        # postprocessing and write tile
                        out = out[0,:classes,:-pad_dims[0],:-pad_dims[1]] if pad_dims!=None else out[0,:classes,:,:]
                        out = out[:,y_left_trim:out.shape[1]-y_right_trim, x_left_trim:out.shape[2]-x_right_trim]
                        out = postprocess(out, segmentationType)
                        bw[y:min([br.Y,y+TILE_SIZE]),x:min([br.X,x+TILE_SIZE]),0,:classes,0] = out.astype(out_dtype)

    except Exception:
        traceback.print_exc()
  
    finally:
        logger.info('Finished Execution')
        # Exit the program
        sys.exit()