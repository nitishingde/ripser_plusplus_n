#!/bin/bash

echo "" > verify.txt
cmake-build-release/ripser++ ./o3_512.point_cloud --format point-cloud --dim 2 --threshold 1 > verify.txt
