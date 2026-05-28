#!/bin/bash

conda deactivate
conda deactivate
conda activate swann_py312

# read and process hrd summary file
outputString=$(python hot_run_from_hrdsummary.py --path $FILEPATH)

# move different lines to different indices
mapfile -t output_lines <<< "$outputString"

# set var names
filesexist=${output_lines[-6]}
stormid=${output_lines[-5]}
legstart=${output_lines[-4]}
legend=${output_lines[-3]}
lat=${output_lines[-2]} # not using right now, might add later ******
lon=${output_lines[-1]} # not using right now, might add later ******

# also add in center location? ******
# add check for winter storms!! ******
if [[ "$filesexist" == 'N' ]]; then 
    echo "python hot_main_run_samurai.py $stormid $legstart $legend > ./output_files/$stormid_$legstart.log"
    python hot_main_run_samurai.py $stormid $legstart $legend > ./output_files/$stormid_$legstart.log
else
    echo "files already processed: $stormid $legstart $legend"
fi
