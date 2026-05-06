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

