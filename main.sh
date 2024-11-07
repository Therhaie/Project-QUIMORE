#!/bin/bash
# echo "Hello World"

NUMBER_OF_SAMPLE=10000
NUMBER_OF_GRID_X=32
NUMBER_OF_GRID_Y=32
LENGHTSCALE_X=0.02
LENGHTSCALE_Y=0.02
VARIANCE_OF_FIELD=1.0
KERNEL_TYPE=rbf

path="$PWD"
path_python_file="$path"/create_data.py

# Construct the command to run the python script

cmd="python3 $path_python_file -N \"$NUMBER_OF_SAMPLE\" -nx \"$NUMBER_OF_GRID_X\" -ny \"$NUMBER_OF_GRID_Y\" -lx \"$LENGHTSCALE_X\" -ly \"$LENGHTSCALE_Y\" -var \"$VARIANCE_OF_FIELD\" -k \"$KERNEL_TYPE\""


# Output the final command for debugging purposes
echo "Final command: $cmd"
echo " "

# Run the command
eval "$cmd"
