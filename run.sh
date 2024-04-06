#!/bin/bash

echo "" > verify.txt
cmake-build-release/hh ./o3_512.point_cloud --dim 2 --threshold 1 > verify.txt
