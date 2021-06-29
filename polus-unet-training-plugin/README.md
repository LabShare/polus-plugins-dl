# UNet Training

Python wrapper based on the work of Thorsten Falk et al. (ImageJ plugin & caffe backend, https://github.com/lmb-freiburg/Unet-Segmentation). Original work at Nature Methods publication: Falk, Thorsten, et al. "U-Net: deep learning for cell counting, detection, and morphometry." Nature methods 16.1 (2019): 67. https://doi.org/10.1038/s41592-018-0261-2

Contact [Vishakha Goyal](mailto:vishakha.goyal@nih.gov) for more information.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes ten input arguments and one output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--borderWeightFactor` | lambda separation | Input | number |
| `--borderWeightSigmaPx` | Sigma for balancing weight function. | Input | number |
| `--foregroundbackgroundgratio` | Foreground/Background ratio | Input | number |
| `--pixelsize` | Input image pixel size (in µm). If pixel size is not provided, it will be read from image metadata. If no metadata is available, model resolution will be used as pixel size.| Input | number |
| `--iterations` | Number of training iterations. | Input | number |
| `--sigma1Px` | Sigma for instance segmentation. | Input | number |
| `--testingImages` | Input testing image collection to be processed by this plugin | Input | collection |
| `--trainingImages` | Input training image collection to be processed by this plugin | Input | collection |
| `--testingLabels` | Input testing mask collection to be processed by this plugin | Input | collection |
| `--trainingLabels` | Input training mask collection to be processed by this plugin | Input | collection |
| `--outDir` | Output collection | Output | genericData |

