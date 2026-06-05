#!/bin/bash

# find all new HRD summary files within past 180 minutes (**modify time**)
hrd_files=$(find ./ingest_dir/hrd_tar -type f -mmin -30) # could make a python script?

mkdir -p ./output_files

# only run code if variable is not empty
if [[ -n "$hrd_files" ]]; then 

    # create variable that contains all files
    mapfile -t hrd_files_arr <<< "$hrd_files"

    # (add check for ingest dir???)

    # run trigger vdm script... just AF flights?
    for ((i=0; i<${#hrd_files_arr[@]}; i++)); do
        echo "${hrd_files_arr[i]}"
        ./trigger_swann_from_hrdsummary.sh "${hrd_files_arr[i]}"
    done

else

    echo "No new files found."

fi