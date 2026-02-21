#!/bin/bash

outputString=$(python hot_run_from_vdm.py)

set $outputString

python hot_main_run_hdobsonly.py $1 $2 $3 N