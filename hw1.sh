#!/bin/bash

if [ $# -ne 2 ]; then
  echo -e "usage:\t./hw1.sh input_file_path output_file_path"
  exit
fi

python3 hw1.py $1 $2
