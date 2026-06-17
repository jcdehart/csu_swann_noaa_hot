#!/bin/bash

# read and process hrd summary file
outputString=$(python hot_run_from_hrdsummary.py --path $1)

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
lat=${output_lines[-2]} # not using right now, might add later ******
lon=${output_lines[-1]} # not using right now, might add later ******

# also add in center location? ******
if [[ "$filesexist" == 'N' && "$tc" == 'TC' && "$dataexist" == 'True' ]]; then 
    echo "python hot_main_run_samurai.py $stormid $stormname $legstart $legend > ./output_files/sam_$stormid_$legstart.log"
    python hot_main_run_samurai.py $stormid $stormname $legstart $legend > ./output_files/sam_$stormid_$legstart.log
elif [[ "$filesexist" == 'N' && "$tc" == 'TC' && "$dataexist" == 'False' ]]; then 
    echo "Not enough input files exist, waiting for more data: $stormid $stormname $legstart $legend"
else
    echo "Not processing files: $stormid $stormname $legstart $legend, files exist: $filesexist, storm type: $tc"
fi
