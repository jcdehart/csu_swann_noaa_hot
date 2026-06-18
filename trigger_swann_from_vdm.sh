#!/bin/bash

outputString=$(python hot_run_from_vdm.py --path $1)

# move different lines to different indices
mapfile -t output_lines <<< "$outputString"

# set var names
dataexist=${output_lines[-9]}
filesexist=${output_lines[-8]}
tc=${output_lines[-7]}
stormid=${output_lines[-6]}
stormname=${output_lines[-5]}
legstart=${output_lines[-4]}
legend=${output_lines[-3]}
lat=${output_lines[-2]} 
lon=${output_lines[-1]} 

if [[ "$filesexist" == 'N' && "$tc" == 'TC' && "$dataexist" == 'True' ]]; then 
    echo "python hot_main_run_hdobsonly.py $stormid $stormname $legstart $legend A --VDMLAT $lat --VDMLON $lon > ./output_files/hdobs_$stormid_$legstart.log"
    python hot_main_run_hdobsonly.py $stormid $stormname  $legstart $legend A --VDMLAT $lat --VDMLON $lon > ./output_files/hdobs_$stormid_$legstart.log
elif [[ "$filesexist" == 'N' && "$tc" == 'TC' && "$dataexist" == 'False' ]]; then 
    echo "Not enough input files exist, waiting for more data: $stormid $stormname $legstart $legend"
else
    echo "Not processing files: $stormid $stormname $legstart $legend, files exist: $filesexist, storm type: $tc"
fi