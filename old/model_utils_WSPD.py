# Model helper functions for the Sfc Wind Reduciton NN Model
# Created by Alex DesRosiers - 7/14/21 for stable release that day

#Import Statements
import numpy as np


#Function to standardize the input data with the characteristics of training set
#Run this function on the training data
def Standardize_Vars(in_arr):
    standardize_input = lambda dat, x, s: (dat - x)/s 

    # Have the mean and standard deviation of the training data available
    trainmean = np.asarray([200046499.2283817,7182789.01306776,-645753.3269913851,3014292758.9564495,266002467462.16727,8733971335.125872,494609278.93076676,6070701316.880631])
    trainstd  = np.asarray([117566663.68993029,69378830.8430621,71655902.56134088,1179201811.3236659,61365894021.80803,3132346008.687273,226402268.7890431,3893984011.534252])
    trainmean = trainmean/100000000 # Kept precision bwith this division to get back to true vales
    trainstd = trainstd/100000000
    out_arr = standardize_input(in_arr,trainmean,trainstd)
    return out_arr
