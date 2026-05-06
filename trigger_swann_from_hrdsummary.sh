#!/bin/bash

# read and process hrd summary file
outputString=$(python hot_run_from_hrdsummary.py --path $FILEPATH)

# grab only the last 5 words (processed, storm code, start time, end time, lat, lon)
parsedString="${outputString[@]: -5}"

# set to numeric variables
set $parsedString

# also add in center location? ******
# add check for winter storms!! ******
if [[ "$1" == 'N' ]]; then 
    echo "python hot_main_run_hdobsonly.py $2 $3 $4 > ./output_files/$2_$3.log"
    python hot_main_run_hdobsonly.py $2 $3 $4 > ./output_files/$2_$3.log
else
    echo "files already processed: $2 $3 $4"
fi
