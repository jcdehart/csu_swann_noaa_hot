# CSU SWANN Workflow

This repository holds the scripts that run the CSU Surface Winds from Aircraft with a Neural Network (SWANN) model. The model is described in [DesRosiers et al. (2025)](https://doi.org/10.1029/2025JH000584). 


## Running SWANN directly from Python scripts

For help running the python scripts directly, please refer to the instructions below.

### NOAA P3 Aircraft

For NOAA aircraft with TDR data, the basic command structure is as follows:
python hot_main_run_samurai.py stormID (e.g., AL05) leg_start (YYYYMMDDHHMM) leg_end (YYYYMMDDHHMM)

example: `python hot_main_run_samurai.py AL05 202408141201 202408141325`

### Air Force Aircraft

For HDOBS-only flights (Air Force or NOAA aircraft), the basic command structure is as follows:
python hot_main_run_hdobsonly.py stormID (e.g., AL05) leg_start (YYYYMMDDHHMM) leg_end (YYYYMMDDHHMM) (new) plane type (air force: A, NOAA: N)

example: `python hot_main_run_hdobsonly.py AL10 202308281059 202308281149 A`

### Test Mode

To test the code, use the following commands:

`python hot_main_run_samurai.py ALXX xxxxxxxxxxxx yyyyyyyyyyyy --MODE test`

`python hot_main_run_hdobsonly.py ALXX xxxxxxxxxxxx yyyyyyyyyyyy A --MODE test`

Output files will be saved to the ./testing/output directory. Expected output files are provided in the ./testing/output_expected directory.


## Running SWANN from shell scripts

The identify_new_files.sh and identify_new_hrdsumm_files.sh each search for new center files (i.e., VDM files for Air Force and HRD Summary files for NOAA flights, respectively) and then run SWANN if output files (an image, a NetCDF file, and a text file) have not yet been created. SWANN is only run for center files that were created within the past 2 hours. 

To make sure they run correctly, define a variable named SWANNHOME with the path to this SWANN directory in a .env file. For example, `export SWANNHOME="/path/to/csu_swann_noaa_hot"`.


## Setup

The input data needs to be organized in the following manner:

```bash
.
├── ingest_dir
│   ├── center_data
│   │   ├── adeck
│   │   │   ├── 2023
│   │   │   ├── ⋮
│   │   │   ├── 2026
│   │   ├── tcvitals
│   │   │   ├── 20181010
│   │   │   ├── ⋮
│   │   │   ├── 20251027
│   │   └── vdm
│   │   │   ├── 2024
│   │   │   ├── ⋮
│   │   │   ├── 2026
│   ├── hdobs
│   │   ├── 2023
│   │   ├── ⋮
│   │   └── 2026
│   ├── hrd_radials
│   │   ├── 2019
│   │   ├── ⋮
│   │   └── 2026
│   └── hrd_tar
│       ├── 2025
│       └── 2026
```


## Required datasets

[NHC A-decks](https://ftp.nhc.noaa.gov/atcf/aid_public/)

HDOBs (2025): [Atlantic](https://www.nhc.noaa.gov/archive/recon/2025/AHONT1/), [East Pacific](https://www.nhc.noaa.gov/archive/recon/2025/AHOPN1/)

[HRD QC'd Radials](https://www.aoml.noaa.gov/ftp/pub/hrd/reasor/emclist/)

[HRD tarballs](https://www.aoml.noaa.gov/ftp/pub/hrd/reasor/realtime_analyses/)

VDMs (2025): [Atlantic](https://www.nhc.noaa.gov/archive/recon/2025/REPNT2/), [East Pacific](https://www.nhc.noaa.gov/archive/recon/2025/REPPN2/)

[tcvitals (atmos)](https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/)


## Recommended environment setup

`conda create --name swann_py312 python=3.12`

`conda activate swann_py312`

`pip install tensorflow pandas matplotlib netcdf4 scipy tf_keras`


## Installing and setting up Julia

Download and install Julia on your machine using the instructions found [here](https://julialang.org/downloads/).

To simplify package installation, this repo comes with Project.toml and Manifest.toml files to recreate the SWANN dependencies. To setup Julia the first time, 1) move to the main SWANN directory and 2) start julia with the project flag.

`julia --project=.`

That will bring up the Julia interactive REPL, which contains a green-colored 'julia>' prompt. To enter the package manager, type ']' (without quotation marks). That will change the prompt to a blue-colored 'pkg>'. Run `instantiate` to download and compile the packages listed in the provided .toml files. Type `st` to see the installed packages. To exit the package manager, hit the 'delete' or 'backspace' key. To exit Julia, just enter `exit()` like you would in Python.


## Installing SAMURAI

To install SAMURAI, follow the LROSE [installation instructions](https://github.com/NCAR/lrose-core/releases) for your specific operating system. If using Homebrew on a Mac, you'll need to install the lrose-core and samurai .rb files separately. Make sure SAMURAI has been added to your path (e.g., `which samurai` returns the expected install)