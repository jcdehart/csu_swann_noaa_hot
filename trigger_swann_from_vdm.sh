#!/bin/bash

outputString=$(python hot_run_from_vdm.py --path $1)

# move different lines to different indices
mapfile -t output_lines <<< "$outputString"

# set var names
filesexist=${output_lines[-7]}
tc=${output_lines[-6]}
stormid=${output_lines[-5]}
legstart=${output_lines[-4]}
legend=${output_lines[-3]}
lat=${output_lines[-2]} 
lon=${output_lines[-1]} 

# add check for winter storms!! ******

if [[ "$filesexist" == 'N' && "$tc" == 'TC' ]]; then 
    echo "python hot_main_run_hdobsonly.py $stormid $legstart $legend A --VDMLAT $lat --VDMLON $lon > ./output_files/$stormid_$legstart.log"
    python hot_main_run_hdobsonly.py $stormid $legstart $legend A --VDMLAT $lat --VDMLON $lon > ./output_files/$stormid_$legstart.log
else
    echo "files already processed: $stormid $legstart $legend"
    echo $filesexist
    echo $tc
fi