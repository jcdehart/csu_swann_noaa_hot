# CSU SWANN Workflow

This repository holds the scripts that run the CSU Surface Winds from Aircraft with a Neural Network (SWANN) model. The model is described in [DesRosiers et al. (2025)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025JH000584). 

For NOAA aircraft with TDR data, the basic command structure is as follows:
python hot_main_run_samurai.py stormID (e.g., AL05) leg_start (YYYYMMDDHHMM) leg_end (YYYYMMDDHHMM)

example: `python hot_main_run_samurai.py AL05 202408141201 202408141325`

For HDOBS-only flights (Air Force or NOAA aircraft), the basic command structure is as follows:
python hot_main_run_hdobsonly.py stormID (e.g., AL05) leg_start (YYYYMMDDHHMM) leg_end (YYYYMMDDHHMM) (new) plane type (air force: A, NOAA: N)

example: `python hot_main_run_hdobsonly.py AL10 202308281059 202308281149 A`
