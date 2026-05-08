#!/bin/bash

conda deactivate
conda deactivate
conda activate swann_py312

outputString=$(python hot_run_from_vdm.py --path $FILEPATH)

set $outputString

echo "python hot_main_run_hdobsonly.py $1 $2 $3 A --VDMLAT $4 --VDMLON $5 > ./output_files/$1_$2.log"

python hot_main_run_hdobsonly.py $1 $2 $3 A --VDMLAT $4 --VDMLON $5 > ./output_files/$1_$2.log
