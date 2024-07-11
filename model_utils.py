# Model helper functions for the Sfc Wind Reduciton Factor NN Model
# Created by Alex DesRosiers - 7/14/21 for stable release that day
# Modified on 6/5/24 for a model to predict inward to r* of 0.3 for 2024 demo

#Import Statements
import numpy as np


#Function to standardize the input data with the characteristics of training set
#Run this function as last step before prediction
def Standardize_Vars(in_arr):
    standardize_input = lambda dat, x, s: (dat - x)/s 

    # Have the mean and standard deviation of the training data available
    trainmean = np.asarray([208260733.15548772,9122344.32929131,-2249677.0113748824,3197555171.286402,266144603523.7468,9341130392.738155,525596005.4661528,4699141504.365173])
    trainstd  = np.asarray([119447962.3632483,70579074.96233349,70216208.70385674,1101969187.1836104,64474224148.556526,3130458640.5860677,220701806.7476198,1985543331.1573339])
    trainmean = trainmean/100000000 # Kept precision with this division to get back to true vales
    trainstd = trainstd/100000000
    out_arr = standardize_input(in_arr,trainmean,trainstd)
    return out_arr
