#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-smp-inference-plugin:${version}