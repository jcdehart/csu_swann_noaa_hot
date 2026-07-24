# Surface Winds from Aircraft with a Neural Network (SWANN) Model Evaluation

This document serves as a user guide for the software used to create the validation datasets and evaluate SWANN against other models. This model is used in real-time as part of a Hurricane Ocean Testbed (HOT) project. For a scientific description of SWANN, please refer to [DesRosiers et al. (2025a)](https://doi.org/10.1029/2025JH000584).

Authors: Alexander J. DesRosiers, Michael M. Bell, and Jennifer C. DeHart

## Summary of SWANN
The Surface Winds from Aircraft with a Neural Network (SWANN) model is a basic feed-forward artificial neural network which predicts the wind ratio (WR) between surface winds and winds at reconnaissance aircraft flight level in a tropical cyclone (TC) environment. The main model creation code is written in the Python programming language (version 3.7.13). Some code is written in the Julia programming language (version 1.8.5) which was employed for its fast performance when generating datasets for training or validating the model. Several different datasets are utilized in this project including the FLIGHT+ (v1.3; [Vigh et al. 2020](http://dx.doi.org/10.5065/D6WS8R93)), TC-DROPS (v1.2; Zawislak et al. 2018), and TC-RADAR (v3k; [Fischer et al. 2022](https://doi.org/10.1175/MWR-D-21-0223.1)). The model uses the following information to predict the WR:
- r*: Radius normalized to radius of maximum winds (km)
- θ*: Angle relative to 0 degree storm motion (degrees, increasing clockwise)
- Flight-level Wind: Aircraft-recorded winds at aircraft flight level (m/s)
- Altitude: Aircraft altitude at time of observation (m)
- Vmax: Most recent estimate of maximum sustained wind speed (kt)
- Storm Forward Speed: Magnitude of estimated forward motion of the TC (m/s)
- RMW: Radius of maximum winds at flight level (km)

The predictions are pointwise. When predictors are obtained for analyzed aircraft wind field, they can be flattened into an array, fed through the neural network, and subsequently reshaped into a surface wind analysis for forecasters. The following code is based on generating and validating the SWANN model with past data with a goal of real-time use. Far greater detail on this model is given in [DesRosiers et al. (2025a)](https://doi.org/10.1029/2025JH000584) with this document serving mainly as a guide to fill in the blanks for users of the model code repository on FigShare ([DesRosiers et al. 2025b](https://doi.org/10.6084/m9.figshare.28148177)).

## Create Validation Datasets and Evaluate SWANN
This section outlines the necessary steps to creating the validation datasets and then validating SWANN:

1. Generate a training dataset from FLIGHT+
2. Generate flight-level wind fields with predictors from TC-RADAR
3. Generate surface wind dataset with TC-DROPS
4. Evaluate SWANN model performance with dropsondes

Each step will have its own dedicated section which describes code in the repository relevant to completion. Some key auxiliary scripts will be described in later sections.

### Step 1: Generate a Training Dataset from FLIGHT+
The main processing for the FLIGHT+ data to create a training dataset is performed with a Julia script titled “Flight+_Mining_WV.jl” where the “WV” stands for working version. The code loops through each of the level 3 data files in the FLIGHT+ dataset to extract aforementioned flight-level predictors as well as the SFMR surface wind data. The code operates by looping through each FLIGHT+ level 3 data file in the directory. The code also uses some external Best Track data to pull general info such as the current and last reported intensity of the storm. The Best Track data used in this code comes from a github repository created by user “ResidentMario” titled “hurdat2”. The Python notebooks in the hurdat2 GitHub repository cleans up the Best Track text data files and returns streamlined CSVs. I have prepared the Best Track data for the FLIGHT+ processing script using this repository. 

The end result of running this script is a NetCDF file which contains surface and flight-level wind data with all observations in a pointwise sequential structure. Subsequent scripts are used to clean up this more raw dataset when doing things such as throwing out bad observations, pruning the training data, or time averaging the observations to smooth them.

### Step 2: Generate Flight-level Wind Fields with Predictors from TC-RADAR
The TC-RADAR dataset is used to construct a backlog of smoothed tail Doppler radar (TDR) derived flight-level wind fields for model evaluation. Originally, this process was performed with an earlier version of TC-RADAR (v3j) and that can be found in the notebook titled “WR_Testing_Dataset_Creation.ipynb”. The more updated code which uses version v3k is in “WR_new_testing_TCRADAR.ipynb”. This version of the dataset grew TC-RADAR, requiring multiple source files. The notebook, written in Julia, now merges the variables from the two files into one continuous set of variables for all past observations. This notebook also makes use of the CSV files processed from Best Track data. The notebook takes the TC-RADAR data, and outputs a flight-level wind field (full and smoothed WN 0+1) with 2 km radial resolution and 1 degree azimuthal resolution. Note that the TC-RADAR analyses are merged over the full course of a TC reconnaissance flight which should cause some temporal smoothing of the wind field and reduce maximum values. Predictors required for the model are assigned at every point in the polar grid space and flattened. The flattened arrays are output in a NetCDF with the necessary dimensions to reshape the arrays when read into future notebooks.

### Step 3: Generate Surface Wind Dataset with TC-DROPS
The TC-DROPS code can be found in the “TCDROPS” folder. The setup is similar to the FLIGHT+ mining in which there is a main Julia processing script (“TCD_Mining_WV.jl”). The data can be requested from either Jun Zhang or Jonathan Zawislak who helped develop the database. The current version of the code parses the dataset to locate surface wind observations from dropsondes within the r* and θ* polar grid. The surface winds are acquired via the WL150 technique which takes the winds over the lowest recorded 150 m of the dropsonde path. These winds are then reduced to the surface via the technique described in Franklin, Black, and Valde (2003). The surface wind validation points are output as a NetCDF file for later use.

### Step 4: Evaluate SWANN Model Performance with Dropsondes
The SFMR has several known issues and biases so we validate the SWANN model with a mostly independent dataset. Data from past hurricane reconnaissance flights offers an opportunity to test the model predictions against one of the closest “ground truths” we have for a surface wind, dropsondes. The basic method for making these comparisons is to get a flight-level wind field from TC-RADAR, project a wind field down to the surface with the SWANN model, and compare historical dropsonde WL150 surface wind measurements to the nearest grid box in a polar grid defined by r* and θ*. The notebook titled “Merged_Data_TCD_HS24_rtheta_Validation.ipynb” does this in Python. The main code which loops through all previous flights is found below the markdown subheading titled “Put dropsondes in for validation”. There are several iterations of code to generate the plots used in the manuscript for binning of the observations and comparing the SWANN model performance to Simplified Franklin.

There was an attempt to use latitude and longitude pairs to co-locate the dropsondes with the SWANN model predictions of surface winds using this same data. That attempt is found in the notebook titled “Merged_Data_TCD_Wind_Pred_HS24_Validation.ipynb”. The errors using this method were found to be significantly larger as compared to the method involving r* and θ*. If switching to this method is desired, a considerable debugging effort would need to be undertaken with this code and the dataset creation code for TC-RADAR as well as TC-DROPS to verify the latitude and longitude information is correct. Regardless, the end of this notebook contains a section of code under the markdown subheading “Work on Vmax Comparisons for the models” which uses only the TC-RADAR and Best Track data to compare storm intensity estimates from the SWANN model to the operationally-determined values. 

## References

DesRosiers, A. J., Bell, M. M., DeHart, J.C., Vigh, J. L., Rozoff, C. M., &Hendricks, E. A. (2025a). Tropical cyclone surface winds from aircraft with a neural network. Journal of Geophysical Research: Machine Learning and Computation, 2, e2025JH000584. https://doi.org/10.1029/2025JH000584

DesRosiers, Alexander; Bell, Michael; DeHart, Jennifer; Vigh, Jonathan; Rozoff, Christopher; Hendricks, Eric (2025b). Tropical Cyclone Surface Wind Reduction with a Neural Network. figshare. Dataset. https://doi.org/10.6084/m9.figshare.28148177.v1

Fischer, M. S., P. D. Reasor, R. F. Rogers, and J. F. Gamache, 2022: An Analysis of Tropical Cyclone Vortex and Convective Characteristics in Relation to Storm Intensity Using a Novel Airborne Doppler Radar Database. Mon. Wea. Rev., 150, 2255–2278, https://doi.org/10.1175/MWR-D-21-0223.1

Vigh, J. L., N. M. Dorst, C. L. Williams, D. P. Stern, E. W. Uhlhorn, B. W. Klotz, J. Martinez, H. E. Willoughby, F. D. Marks, Jr., D. R. Chavas, 2020: FLIGHT+: The Extended Flight Level Dataset for Tropical Cyclones (Version 1.3). Tropical Cyclone Data Project, National Center for Atmospheric Research, Research Applications Laboratory, Boulder, Colorado. http://dx.doi.org/10.5065/D6WS8R93

Zawislak, J., Nguyen, L., Paltz, E., Young, K., Voemel, H., & Hock, T. (2018). Development and applications of a long‐term, global tropicalcyclone dropsonde dataset. In 33rd Conference on Hurricanes and Tropical Metorology. Amer. Meteor. Soc.
