#!/bin/bash

# find all new VDM center files within past 180 minutes (**modify time**)
af_files=$(find ./ingest_dir/center_data/vdm -type f -mmin -180 -name "*NHC*") # could make a python script?

mkdir -p ./output_files

# only run code if variable is not empty
if [[ -n "$af_files" ]]; then 

    # create variable that contains all files
    mapfile -t af_files_arr <<< "$af_files"

    # (add buffer time to ensure hdobs files present? or add check for ingest dir???)

    # run trigger vdm script... just AF flights?
    for ((i=0; i<${#af_files_arr[@]}; i++)); do
        echo "${af_files_arr[i]}"
        ./trigger_swann_from_vdm.sh "${af_files_arr[i]}"
    done

else

    echo "No new files found."

fi
