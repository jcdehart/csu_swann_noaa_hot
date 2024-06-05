# Model helper functions for the Sfc Wind Reduciton Factor NN Model
# Created by Alex DesRosiers - 7/14/21 for stable release that day
# Modified on 2/19/24 for new model predicting reduction factor with 2D loss

#Import Statements
import numpy as np


#Function to standardize the input data with the characteristics of training set
#Run this function as last step before prediction
def Standardize_Vars(in_arr):
    standardize_input = lambda dat, x, s: (dat - x)/s 

    # Have the mean and standard deviation of the training data available
    trainmean = np.asarray([216557529.7243551,9279799.256318778,-2232538.9073133655,3229804561.2477913,267360379391.0892,9379684131.37053,526420976.9134795,4646892386.636683])
    trainstd  = np.asarray([116263928.21781428,70657885.15638363,70116790.86365096,1103244410.4686546,64366488432.27685,3132965881.24685,221397043.51362792,1965782271.3637643])
    trainmean = trainmean/100000000 # Kept precision with this division to get back to true vales
    trainstd = trainstd/100000000
    out_arr = standardize_input(in_arr,trainmean,trainstd)
    return out_arr
