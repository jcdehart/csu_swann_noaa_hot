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

## Setup

The input data should be organized in the following manner:

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

## Needed datasets

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