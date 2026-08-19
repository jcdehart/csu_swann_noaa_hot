# Surface Winds from Aircraft with a Neural Network (SWANN) Model User Guide

This document serves as a basic user guide for creating the SWANN model. This model is used in real-time as part of a Hurricane Ocean Testbed (HOT) project. For a scientific description of SWANN, please refer to [DesRosiers et al. (2025a)](https://doi.org/10.1029/2025JH000584).

Authors: Alexander J. DesRosiers, Michael M. Bell, and Jennifer C. DeHart

## Summary of Current Version
The Surface Winds from Aircraft with a Neural Network (SWANN) model is a basic feed-forward artificial neural network which predicts the wind ratio (WR) between surface winds and winds at reconnaissance aircraft flight level in a tropical cyclone (TC) environment. The main model creation code is written in the Python programming language (version 3.7.13) as several scripts and notebooks which are described in subsequent sections for users. Some code is written in the Julia programming language (version 1.8.5) which was employed for its fast performance when generating datasets for training or validating the model. Several different datasets are utilized in this project including the FLIGHT+ (v1.3; [Vigh et al. 2020](http://dx.doi.org/10.5065/D6WS8R93)), TC-DROPS (v1.2; [Zawislak et al. 2018](https://ams.confex.com/ams/33HURRICANE/meetingapp.cgi/Paper/339581), obtained from the authors), and TC-RADAR (v3k; [Fischer et al. 2022](https://doi.org/10.1175/MWR-D-21-0223.1)). The model uses the following information to predict the WR:
- r*: Radius normalized to radius of maximum winds (unitless)
- θ*: Angle relative to 0 degree storm motion (degrees, increasing clockwise)
- Flight-level Wind: Aircraft-recorded winds at aircraft flight level (m/s)
- Altitude: Aircraft altitude at time of observation (m)
- Vmax: Most recent estimate of maximum sustained wind speed (kt)
- Storm Forward Speed: Magnitude of estimated forward motion of the TC (m/s)
- RMW: Radius of maximum winds at flight level (km)

The predictions are pointwise. When predictors are obtained for analyzed aircraft wind field, they can be flattened into an array, fed through the neural network, and subsequently reshaped into a surface wind analysis for forecasters. The following code is based on generating and validating the SWANN model with past data with a goal of real-time use. Far greater detail on this model is given in [DesRosiers et al. (2025a)](https://doi.org/10.1029/2025JH000584) with this document serving mainly as a guide for the model code repository on FigShare ([DesRosiers et al. 2025b](https://doi.org/10.6084/m9.figshare.28148177)).

## Train and Tune SWANN 
This section outlines the necessary steps to create SWANN from the existing datasets and code base. Information on training and tuning SWANN are provided below:

1. Generate and evaluate the tuned SWANN model
2. Tune the neural network for optimized performance

### 1: Generate and Evaluate SWANN
The notebook titled “SWANN_Model_Generation” contains the Python code to train and begin evaluation of the tuned version of the SWANN model. The FLIGHT+ data, found in the Polar_Norm_Winds_All NetCDF file provided on Figshare, is read in and pre-processed. Several conditions described in the manuscript are implemented, including a bound on WR from 0.6 to 1.4, and removing any NaN observations or observations with flight-level winds below 15 m/s. The data are smoothed to be a 10-second average of wind speed both at flight level and the surface. The averaged wind is only included if 6 out of the 10 seconds in the averaging period have a valid wind speed observation. Data from flights into Hurricane Dorian (2019) are excluded due to documented concerns with the SFMR during these missions. Once undesirable data has been removed, the validation and testing sets, which include all data collected in Hurricanes Michael (2018) and Matthew (2016), respectively, are extracted.

There are also functions designed to mimic the current operational Franklin wind reduction procedures for comparison. More details on this “Simplified Franklin” technique are in [DesRosiers et al. (2025a)](https://doi.org/10.1029/2025JH000584). Once the data have been processed, they are normalized based on the mean and standard deviation values of each of the model predictors. The function “Standardize_Vars” is introduced here to provide a blueprint for how to use the characteristics of the training data with the SWANN model, without loading and pre-processing all of the data. 

The loss function, referred to in the script as “MSE_SCL_loss2d” uses both the predictand and supplemental information and is described in [DesRosiers et al. (2025a)](https://doi.org/10.1029/2025JH000584) in Section 2.3. In this case, the predictand is the WR while the supporting information is the actual magnitude of the SFMR wind. The supporting information is utilized by the shifted cubic loss portion of the function which weights strong wind observations more heavily. Details about the custom loss function are provided in the manuscript. The basic mean squared error (MSE) must also be written as a custom function to deal with the 2D predictand array required for this custom loss configuration. The function is provided in the code and titled “mse_ytrue2D”.

Once the model is run, there are plenty of extraneous blocks of code to evaluate how the model works. Towards the end of this notebook, there are some examples of code that start to utilize the TC-RADAR and TC-DROPS datasets described in the SWANN validation document. However, these are unimportant at this stage as a more detailed methodology has been developed in subsequent notebooks. Some of the figures in this notebook are highlighted in the manuscript where a more lengthy discussion is provided. This notebook also saves the model before evaluating in the first block of code below the markdown heading titled “Evaluate the Model”. This block of code demonstrates how to save the model architecture and weights as well as how to load them from these files later. Performance is evaluated on  the training, validation, and testing sets to make sure there is improvement over the current operational benchmark of the Simplified Franklin method.

This particular methodology for training a neural network came from a long trial and error period in which many different methods were evaluated. To understand the performance of this method in reference to different attempts, the notebook titled “HS24_NN_Comparison” was developed. The notebook trains several different basic neural network configurations for surface wind predictions. A table in the manuscript describes the defining differences between each of these methods and the basis for their nomenclature. This notebook can also be skipped for a more streamlined user who wants to simply develop the SWANN model with the chosen method described in the manuscript.

### 2: Tune the Neural Network for Optimized Performance
To test potential hyperparameter choices that may affect model performance, the KerasTuner package is employed. The script “Model_Tuner2D.py” helps span potential choices for the number of hidden layers, number of nodes in each hidden layer, ridge regression coefficient, learning rate, and dropout rate. Other settings such as the activation function, optimizer, and loss function are held static. This code was adapted from code provided by Marybeth Arcodia for use of the KerasTuner package. 

The end result of this code will be the output text which indicates the recommended optimal configuration of the hyperparameters that were varied in the search. Note that the section near “Trying Arcodia Best HP Method” has not been debugged so the previous hardcoded text in the output is the correct one to reference for tuning results. This step can be skipped if you’d like to just move on and generate the model based on this version. The output from the tuning script recommends the following hyperparameters:

- Number of hidden layers: 1
- Number of nodes in hidden layer: 20
- Learning Rate: 5.0912440719168127e-05
- Ridge regression coefficient: 0.2
- Dropout rate: 0.4

## Use of the Current Model
The current version of the model can be found on Figshare. The model architecture is in the “HS24_SCL_2DNN_model_v2.json” file and the model weights are found in the “HS24_SCL_2DNN_model_v2.h5” file. Steps to load the model are found in several of the evaluation scripts above. If using the model externally, you will need mean and standard deviation values for the training dataset to standardize the inputs to the model. An example helper script shows how to store that information for model use (“HS24_v2_model_utils.py”). 

An example of how SWANN is loaded in this repository is provided here:

```python
# load json and create model
json_file = open(ml_dir+json_fn, 'r')
loaded_model_json = json_file.read()
json_file.close()
nn_model = model_from_json(loaded_model_json)

# load weights into new model
nn_model.load_weights(ml_dir+ml_file)
print("Loaded model from disk")

# make prediction with the neural net
predict = nn_model.predict(x_data)
```

## Closing Thoughts
Ideally, this guide paired with the manuscript will help make some sense of what each of these scripts are doing. It should also serve to help a user create the SWANN model on their own should they desire to. Due to space constraints, the datasets are omitted in the repository, but should be available for download. Reach out to Alex DesRosiers for any particular questions about the code, guide, or manuscript.

## References

DesRosiers, A. J., Bell, M. M., DeHart, J.C., Vigh, J. L., Rozoff, C. M., &Hendricks, E. A. (2025a). Tropical cyclone surface winds from aircraft with a neural network. Journal of Geophysical Research: Machine Learning and Computation, 2, e2025JH000584. https://doi.org/10.1029/2025JH000584

DesRosiers, Alexander; Bell, Michael; DeHart, Jennifer; Vigh, Jonathan; Rozoff, Christopher; Hendricks, Eric (2025b). Tropical Cyclone Surface Wind Reduction with a Neural Network. figshare. Dataset. https://doi.org/10.6084/m9.figshare.28148177.v1

Fischer, M. S., P. D. Reasor, R. F. Rogers, and J. F. Gamache, 2022: An Analysis of Tropical Cyclone Vortex and Convective Characteristics in Relation to Storm Intensity Using a Novel Airborne Doppler Radar Database. Mon. Wea. Rev., 150, 2255–2278, https://doi.org/10.1175/MWR-D-21-0223.1

Vigh, J. L., N. M. Dorst, C. L. Williams, D. P. Stern, E. W. Uhlhorn, B. W. Klotz, J. Martinez, H. E. Willoughby, F. D. Marks, Jr., D. R. Chavas, 2020: FLIGHT+: The Extended Flight Level Dataset for Tropical Cyclones (Version 1.3). Tropical Cyclone Data Project, National Center for Atmospheric Research, Research Applications Laboratory, Boulder, Colorado. http://dx.doi.org/10.5065/D6WS8R93

Zawislak, J., Nguyen, L., Paltz, E., Young, K., Voemel, H., & Hock, T. (2018). Development and applications of a long‐term, global tropicalcyclone dropsonde dataset. In 33rd Conference on Hurricanes and Tropical Metorology. Amer. Meteor. Soc.
