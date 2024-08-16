# Model helper functions for the Sfc Wind Reduciton Factor NN Model
# Created by Alex DesRosiers - 7/14/21 for stable release that day
# Modified on 6/5/24 for a model to predict inward to r* of 0.3 for 2024 demo
# Modified on 7/18/24 to account for 600 to 900 mb pressure levels in a new model (v2)

#Import Statements
import numpy as np


#Function to standardize the input data with the characteristics of training set
#Run this function as last step before prediction
def Standardize_Vars(in_arr):
    standardize_input = lambda dat, x, s: (dat - x)/s 

    # Have the mean and standard deviation of the training data available
    trainmean = np.asarray([207122776.2193164,9026046.992487332,-2067238.5593969964,3162731722.776655,264507987393.2525,9214747690.399755,518016318.67406046,4677333121.855429])
    trainstd  = np.asarray([119279476.74477464,70963947.68521449,69845365.84024948,1100503807.6582718,62943084141.181984,3152651002.4613976,225257228.24457613,2008245027.708318])
    trainmean = trainmean/100000000 # Kept precision with this division to get back to true vales
    trainstd = trainstd/100000000
    out_arr = standardize_input(in_arr,trainmean,trainstd)
    return out_arr
