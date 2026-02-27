#!/bin/bash -l 
# NOTE the -l flag! 
# 
#SBATCH -J hot_erin
#SBATCH -o out.txt
#SBATCH -e err.txt
#SBATCH -p all

for i in /bell-scratch/jcdehart/hot_operational/realtime/ingest_dir/center_data/vdm/2025/*NHC.202508*; do

    outputString=$(python hot_run_from_vdm.py $i)

    set $outputString

    echo "python hot_main_run_hdobsonly.py $1 $2 $3 A"

    python hot_main_run_hdobsonly.py $1 $2 $3 A
    
done