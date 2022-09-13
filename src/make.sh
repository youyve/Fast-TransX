#!/bin/bash

if [ ! -d "dataset_lib" ]; then
  mkdir "dataset_lib"
fi

g++ ./base/Base.cpp -fPIC -shared -o ./dataset_lib/train_dataset_lib.so -pthread -O3 -march=native -std=c++11
