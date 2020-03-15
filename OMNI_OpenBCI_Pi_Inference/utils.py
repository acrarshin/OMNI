import numpy as np
import scipy.signal
import scipy.io
import pickle
from glob import glob
import os
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import TensorDataset,DataLoader
from torch.autograd import Variable

def custom_resample(ECG,fs):
    modified_ECG = []
    for i in range(int(len(ECG) * 500/fs)):
        modified_ECG.append(ECG[int(fs/500*i)].astype(float))
    return modified_ECG

def peak_correction(peak_locs,ecg_records):
    
    mod_peak_locs = []
    ecg_records = ecg_records.cpu().numpy()
    for j in range(len(peak_locs)): 
        mod_peak_locs.append(np.asarray([peak_locs[j][i] - 37 + np.argmax(ecg_records[j,0,peak_locs[j][i]-37:peak_locs[j][i]+37]) for i in range(len(peak_locs[j])) if(peak_locs[j][i]>37 and peak_locs[j][i]<5000-37)]))
    return mod_peak_locs
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
def peak_finder(y_pred_array,x_test):
   
    fs_ = 500 
    peak_locs = []
    for i in range(y_pred_array.shape[0]):
        peak_locs.append(scipy.signal.find_peaks(-y_pred_array[i,:],distance = 120,height = -0.4, prominence = 0.035)[0])
        peak_locs[i] = peak_locs[i][(peak_locs[i] >= 0.5*fs_) & (peak_locs[i] <= 9.5*fs_)]
    modified_peak_locs = peak_correction(peak_locs,x_test)
    return modified_peak_locs

def compute_heart_rate(r_peaks):
    fs_ = 500
    r_peaks = r_peaks[(r_peaks >= 0.5*fs_) & (r_peaks <= 9.5*fs_)]
    return round( 60 * fs_ / np.mean(np.diff(r_peaks)))
        
