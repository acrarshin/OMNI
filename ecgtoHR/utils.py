import torch
import torch.nn as nn
import torch.nn.functional as functional

import numpy as np
import pandas as pd
import scipy.signal
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from tqdm import tqdm
import wfdb as wf

def dist_transform(window_size, ann):
    
    """ Compute distance transform of Respiration signaal based on breath positions
    Arguments:
        window_size{int} -- Window Length  
        ann{ndarray} -- The ground truth R-Peaks
    Returns:
       ndarray -- transformed signal
    """

    length = window_size
    transform = []

    sample = 0
    if len(ann) == 0:
        return None

    if len(ann) ==1:
        for i in range(window_size):
            transform.append(abs(i-ann[sample]))
    else:
        for i in range(window_size):

            if sample+1 == len(ann):
                for j in range(i,window_size):

                    transform.append(abs(j - nextAnn))
                break
            prevAnn = ann[sample]
            nextAnn = ann[sample+1]
            middle = int((prevAnn + nextAnn )/2) 
            if i < middle:
                transform.append(abs(i - prevAnn))
            elif i>= middle:
                transform.append(abs(i- nextAnn))
            if i == nextAnn:
                sample+=1

    transform = np.array(transform)
    minmaxScaler = MinMaxScaler()
    transform = minmaxScaler.fit_transform(transform.reshape((-1,1)))
    return transform

def getWindow(all_paths):
    
    """ Windowing the ECG and its corresponding Distance Transform
    Arguments:
        all_paths{list} -- Paths to all the ECG files
    Returns:
        windowed_data{list(ndarray)},windowed_beats{list(ndarray)} -- Returns winodwed ECG and windowed ground truth
    """

    windowed_data = []
    windowed_beats = []
    count = 0
    count1 = 0
    
    for path in tqdm(all_paths):
        
        ann    = wf.rdann(path,'atr')
        record = wf.io.rdrecord(path)
        beats  = ann.sample
        labels = ann.symbol
        len_beats = len(beats)
        data = record.p_signal[:,0]

        ini_index = 0
        final_index = 0
        ### Checking for Beat annotations
        non_required_labels = ['[','!',']','x','(',')','p','t','u','`',"'",'^','|','~','+','s','T','*','D','=','"','@']
        for window in range(len(data) // 3600):
            count += 1
            for r_peak in range(ini_index,len_beats):
                if beats[r_peak] > (window+1) * 3600:
                    final_index = r_peak
                    #print('FInal index:',final_index)
                    break
            record_anns = list(beats[ini_index: final_index])
            record_labs = labels[ini_index: final_index]
            to_del_index = []
            for actual_lab in range(len(record_labs)):
                for lab in range(len(non_required_labels)):
                    if(record_labs[actual_lab] == non_required_labels[lab]):
                        to_del_index.append(actual_lab)
            for indice in range(len(to_del_index)-1,-1,-1):
                del record_anns[to_del_index[indice]]
            windowed_beats.append(np.asarray(record_anns) - (window) * 3600)
            windowed_data.append(data[window * 3600 : (window+1) * 3600])
            ini_index = final_index

    return windowed_data,windowed_beats

def testDataEval(model, loader, criterion):
    
    """Test model on dataloader
    
    Arguments:
        model {torch object} -- Model   
        loader {torch object} -- Data Loader  
        criterion {torch object} -- Loss Function
    Returns:
        float -- total loss
    """

    model.eval()
    
    with torch.no_grad():
        
        total_loss = 0
        
        for (x,y) in loader:
            
            ecg,BR = x.unsqueeze(1).cuda(),y.unsqueeze(1).cuda()
            BR_pred = model(ecg)
            loss = criterion(BR_pred, BR)
            total_loss += loss
            
    return total_loss


def save_model(exp_dir, epoch, model, optimizer,best_dev_loss):

    """ save checkpoint of model 
    
    Arguments:
        exp_dir {string} -- Path to checkpoint
        epoch {int} -- epoch at which model is checkpointed
        model -- model state to be checkpointed
        optimizer {torch optimizer object} -- optimizer state of model to be checkpoint
        best_dev_loss {float} -- loss of model to be checkpointed
    """

    out = torch.save(
        {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir':exp_dir
        },
        f=exp_dir + '/best_model.pt'
    )

def findValleys(signal, prominence = 10, is_smooth = True , distance = 10):
    
    """ Return prominent peaks and valleys based on scipy's find_peaks function """
    smoothened = smooth(-1*signal)
    valley_loc = scipy.signal.find_peaks(smoothened, prominence= 0.07)[0]
    
    return valley_loc