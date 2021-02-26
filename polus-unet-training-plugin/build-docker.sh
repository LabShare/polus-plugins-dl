#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-unet-training-plugin:${version}