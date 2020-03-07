import numpy as np
import scipy.signal
import scipy.io
from sklearn.preprocessing import StandardScaler
import pickle
from glob import glob
import os
import pyedflib
from tqdm import tqdm
from network import IncUNet

import pandas as pd
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


def obtain_ecg_record(path_records,fs,window_size = 5000):
    
    scaler = StandardScaler()
    ecg_records = []
    actual_ecg_windows = []           
    for path in path_records:
        ECG = scipy.io.loadmat(path)['ecg']
        if(ECG.shape[0] != 1):
            ECG = ECG[:,0]
        else:
            ECG = ECG[0,:]
        ECG = np.asarray(custom_resample(ECG,fs))
        ### Scaling the ECG for every 5000 records(Speed vs Accuracy)
        ecg_windows = []
        for record_no in range(len(ECG)//(window_size-500)):
            ecg_windows.append(scaler.fit_transform(ECG[4500*record_no : 4500*record_no + window_size].reshape(-1,1)))
        for record_no in range(len(ECG)//(window_size)):
            actual_ecg_windows.append(scaler.fit_transform(ECG[5000*record_no : 5000*record_no + window_size].reshape(-1,1)))       
        ecg_records.append(np.asarray(ecg_windows))
    initial = np.zeros((1,5000))
    for ecg_record in ecg_records:       
        if(ecg_record[-1].shape[0] != 5000):
            ecg_record[-1] = np.vstack((ecg_record[-1],np.zeros(5000 - ecg_record[-1].shape[0]).reshape(-1,1)))
        for window_no in range(len(ecg_record)):            
            initial = np.vstack((initial,ecg_record[window_no][:,0].reshape(1,-1)))
    
    ecg_records = initial[1:,:]
    actual_ecg_windows = np.asarray(actual_ecg_windows)[:,:,0]
    return ecg_records,actual_ecg_windows

def load_model_CNN(SAVED_MODEL_PATH,test_loader,device,batch_len,window_size):
        
        C,H,W = 1,1,5000
        loaded_model = IncUNet(in_shape=(C,H,W))    
        loaded_model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location = lambda storage, loc: storage, pickle_module=pickle))
        loaded_model.to(device)
        loaded_model.eval()
        print("-------- Evaluation --------")
        loaded_model.eval()
        net_test_loss = 0   
        pred_peaks = []
        peak_array = np.array([1])
        with torch.no_grad():
            pred_peak = []
            for step,(x) in tqdm(enumerate(test_loader)):
                x = Variable(x[0].to(device))
                y_predict_test = loaded_model(x)
                if(str(device)[:4] == 'cuda'):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
                    y_predict_test = y_predict_test.cpu().numpy()
                    y_predict_test = y_predict_test[:,0,:]
                predicted_peak_locs = peak_finder(y_predict_test,x)
                predicted_peak_locs = np.asarray(predicted_peak_locs) 
                for i in range(len(predicted_peak_locs)):
                    predicted_peak_locs_new = predicted_peak_locs[i][(predicted_peak_locs[i] >= 0.5*500) & (predicted_peak_locs[i] <= 9.5*500)]
                    pred_peaks.append(np.asarray(predicted_peak_locs_new + i*(window_size - 500) + step * batch_len * (window_size - 500)).astype(int))
            for i in range(len(pred_peaks)):
                peak_array = np.hstack((peak_array,pred_peaks[i])) 
            actual_peak_locs = peak_array[1:]  ### As peak locs were initialized with one. 
        return actual_peak_locs                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     

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
        
