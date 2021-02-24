#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-unet-testing-plugin:${version}