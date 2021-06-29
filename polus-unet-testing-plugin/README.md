# UNet Testing

Bfio implementation of python wrapper by Jan Lause: https://github.com/jlause/unet-segmentation-python. Python wrapper based on the work of Thorsten Falk et al. (ImageJ plugin & caffe backend, https://github.com/lmb-freiburg/Unet-Segmentation). Original work at Nature Methods publication: Falk, Thorsten, et al. "U-Net: deep learning for cell counting, detection, and morphometry." Nature methods 16.1 (2019): 67. https://doi.org/10.1038/s41592-018-0261-2

Contact [Vishakha Goyal](mailto:vishakha.goyal@nih.gov) for more information.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes five input arguments and one output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--weightsfilename` | Weights file for testing. | Input | string |
| `--inpDir` | Input image collection to be processed by this plugin. | Input | collection |
| `--filePattern` | File name pattern to filter data. | Input | string |
| `--pixelsize` | Input image pixel size (in µm). If pixel size is not provided, it will be read from image metadata. If no metadata is available, model resolution will be used as pixel size. | Input | number |
| `--weights` | Weights path for testing. | Input | generic data collection |
| `--outDir` | Output collection | Output | collection |

