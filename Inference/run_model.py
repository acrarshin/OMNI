import argparse
import h5py 
import numpy as np
import os
from functools import reduce
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import sys
from pyqtgraph.Qt import QtGui, QtCore

import torch
from torch.utils.data import TensorDataset,DataLoader

from PyQT_Plot import create_dashboard
from preprocess_data import data_read,windowing_and_resampling_hr,windowing_and_resampling_br
from utils import load_model_HR,load_model_BR,compute_heart_rate

def main(args):
    
    preprocessed_patient_data = data_read(args)
    print('-------- Data Acquisition Complete --------')
    windowed_patient_overlap,windowed_patient = windowing_and_resampling_hr(preprocessed_patient_data)
    print('-------- Pre-processing Complete for HR---------')
    windowed_patient_overlap_br = windowing_and_resampling_br(preprocessed_patient_data)
    print('-------- Pre-processing Complete for BR---------')

    ###
    patient_ecg = np.asarray(windowed_patient_overlap['ecg'][0][:60])
    actual_ecg_windows = np.asarray(windowed_patient['ecg'][0][:60])
    
    patient_ecg_br = np.asarray(windowed_patient_overlap_br['ecg'][0][:60])
    ###

    batch_len = 32
    batch_len_br = 1
    window_size = 5000

    patient_ecg = torch.from_numpy(patient_ecg).view(patient_ecg.shape[0],1,patient_ecg.shape[1]).float()
    input_ecg = TensorDataset(patient_ecg)
    testloader = DataLoader(input_ecg,batch_len)

    patient_ecg_br = torch.from_numpy(patient_ecg_br).view(patient_ecg_br.shape[0],1,patient_ecg_br.shape[1]).float()
    input_ecg_br = TensorDataset(patient_ecg_br)
    testloader_br = DataLoader(input_ecg_br,batch_len_br)

    SAVED_HR_MODEL_PATH = args.saved_hr_model_path
    SAVED_BR_MODEL_PATH = args.saved_br_model_path
    device = args.device
    
    ecg_peak_locs = load_model_HR(SAVED_HR_MODEL_PATH,testloader,device,batch_len,window_size)     
    br_peak_locs = load_model_BR(SAVED_BR_MODEL_PATH,testloader_br,device,batch_len,window_size)

    ### Finding Stored Paths
    save_dir = args.save_dir
    if not(os.path.isdir(save_dir)):
        os.mkdir(save_dir)

    save_path =  save_dir + '/r_peaks_patient_' + str(args.patient_no) + '.csv'

    all_hr = []
    initial_hr = len([peak for peak in list(ecg_peak_locs) if peak < 5000 * 6])
    
    for i in range(patient_ecg.shape[0]):
        all_hr.append( len([peak for peak in list(ecg_peak_locs) if peak > i * 2500 and peak < (i * 2500 ) + 5000 * 6 ]))
    unique = np.unique(np.asarray(all_hr))
    peak_no = np.linspace(1,len(ecg_peak_locs),len(ecg_peak_locs)).astype(int)
    peak_no = peak_no.reshape(-1,1)
    ecg_peak_locs = ecg_peak_locs.reshape(-1,1) 
    ecg_peak_locs = np.hstack((peak_no,ecg_peak_locs))

    pd.DataFrame(ecg_peak_locs).to_csv(save_path , header=None, index=None)  
    print('-------- R Peaks Saved --------')

    all_br = []
    initial_br = len([peak for peak in list(br_peak_locs) if peak < 1250 * 6])
    for i in range(patient_ecg.shape[0]):
        all_br.append( len([peak for peak in list(br_peak_locs) if peak > i * 625 and peak < (i * 625 ) + 1250 * 6 ]))
        # all_br.append( len([peak for peak in list(br_peak_locs) if peak > i * 2500 and peak < (i * 2500 ) + 5000 * 6 ]))

    i = 1
    scatter_peak = []
    scatter_peak_1 = []
    ecg_point = []
    ecg_point_1 = []
    k = 0
    hr = []
    peak_locs = ecg_peak_locs[:,1]
    for j in range(len(peak_locs)):     
        if(peak_locs[j] < 5000*i):
            scatter_peak.append(peak_locs[j]-5000*(i-1))
            if(i< len(actual_ecg_windows)):
                ecg_point.append(actual_ecg_windows[i-1,scatter_peak[k]])
                k = k+1                         
        elif(peak_locs[j] >= 5000*i):
            scatter_peak_1.append(np.asarray(scatter_peak))
            hr.append(compute_heart_rate(scatter_peak_1[i-1]))
            ecg_point_1.append(np.asarray(ecg_point))                     
            scatter_peak = []
            ecg_point = []
            i = i+1
            scatter_peak.append(peak_locs[j]-5000*(i-1))
            k = 0
            if(i< len(actual_ecg_windows)):
                ecg_point.append(actual_ecg_windows[i-1,scatter_peak[k]])
                k = k+1
    import pdb;pdb.set_trace()
    if(args.viewer):
        create_dashboard(actual_ecg_windows,scatter_peak_1,all_hr,all_br)
    
    
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dir',help = 'Path to all the records')
    parser.add_argument('--saved_hr_model_path',help = 'Path to saved Heart rate model')
    parser.add_argument('--saved_br_model_path',help = 'Path to saved breathing rate model')
    parser.add_argument('--patient_no',default = 8,type = int,help = 'Patient used for testing')
    parser.add_argument('--device',default = 'cuda', help = 'cpu/cuda')
    parser.add_argument('--save_dir',default = 'saved_models/',help = 'Directory used for saving')
    parser.add_argument('--viewer',default = 0,type = int, help = 'To view ECG plot: 1, else: 0')
    
    args = parser.parse_args()

    main(args)
    
