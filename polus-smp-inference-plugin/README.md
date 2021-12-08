# SMP Inference Plugin

This WIPP plugin serves as the inference plugin for models trained using the `polus-smp-training-plugin`. The repository consists of two plugin manifests, one each for binary and cellpose type segmentation but they both use the same codebase. This is because the type of output is different for both types of segmentation. Binary segmetantation outputs an image collection whereas cellpose segmentation outputs a genericData collection.


Contact [Gauhar Bains](mailto:gauhar.bains@labshare.org) or [Nick Schaub](mailto:nick.schaub@labshare.org) for more information.
For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes 2 input argument and 1 output argument:

| Name          | Description             | I/O    | Type   | Required | Default |
|---------------|-------------------------|--------|--------|----------|---------|
| `--Pattern`  | Filename pattern used to separate data | Input | string | Yes | - |
| `--inpDir`  | Input image collection to be processed by this plugin | Input | collection | Yes | - |
| `--modelPath`  | path to model | genericData | genericData | Yes | - |
| `--outDir`  | Output collection | Output | collection/genericData | Yes | - |

