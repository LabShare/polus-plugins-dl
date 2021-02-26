# UNet Training

This WIPP plugin does things, some of which involve math and science. There is likely a lot of handwaving involved when describing how it works, but handwaving should be replaced with a good description. However, someone forgot to edit the README, so handwaving will have to do for now. Contact [Vishakha Goyal](mailto:vishakha.goyal@nih.gov) for more information.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes one input argument and one output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--borderWeightFactor` | lambda separation | Input | number |
| `--borderWeightSigmaPx` | Sigma for balancing weight function. | Input | number |
| `--foregroundbackgroundgratio` | Foreground/Background ratio | Input | number |
| `--pixelsize` | Input image pixel size. | Input | number |
| `--sigma1Px` | Sigma for instance segmentation. | Input | number |
| `--testing_images` | Input testing image collection to be processed by this plugin | Input | collection |
| `--training_images` | Input training image collection to be processed by this plugin | Input | collection |
| `--output_directory` | Output collection | Output | genericData |

