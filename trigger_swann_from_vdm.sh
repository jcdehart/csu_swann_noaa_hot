#!/bin/bash

outputString=$(python hot_run_from_vdm.py)

set $outputString

echo "python hot_main_run_hdobsonly.py $1 $2 $3 A --VDMLAT $4 --VDMLON $5"

python hot_main_run_hdobsonly.py $1 $2 $3 A --VDMLAT $4 --VDMLON $5
