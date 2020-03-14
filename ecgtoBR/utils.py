import torch
import torch.nn as nn
import torch.nn.functional as functional

import numpy as np
import pandas as pd
import scipy.signal
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
            
            ecg,BR = x.cuda(),y.cuda()
            BR_pred = model(ecg)
            loss = criterion(BR_pred, BR)
            total_loss += loss
            
    return total_loss

def smooth(signal,window_len=50):
    """Compute moving average of specified window length
    
    Arguments:
        signal {ndarray} -- signal to smooth
    
    Keyword Arguments:
        window_len {int} -- size of window over which average is to be computed (default: {50})
    
    Returns:
        ndarray   -- smoothed signal
    """
    
    y = pd.DataFrame(signal).rolling(window_len,center = True, min_periods = 1).mean().values.reshape((-1,))
    return y

def findValleys(signal, prominence = 0.07):
    """Find valleys of distance transform to estimate breath positions   
    Arguments:
        signal {ndarray} -- transform to get breath positions
    
    Keyword Arguments:
        prominence {int} -- threshold prominence to detect peaks (default: {0.07})
    
    Returns:
        ndarray -- valley locations in signal
    """
    smoothened = smooth(-1*signal)
    valley_loc = scipy.signal.find_peaks(smoothened, prominence= prominence)[0]
    
    return valley_loc

def getBR(signal, model):
    """ Get Breathing Rate after passing ECG through Model
    
    Arguments:
        signal {torch tensor} -- input ECG signal
        model  -- ECG to BR model
    
    Returns:
        ndarray -- position of predicted valley and corresponding predicted transform
    """
    
    model.eval()
    with torch.no_grad():
        transformPredicted = model(signal)
    transformPredicted = transformPredicted.cpu().numpy().reshape((-1,))
    valleys = findValleys(transformPredicted)
    return valleys, transformPredicted

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

def dist_transform(signal, ann):

    """ Compute distance transform of Respiration signaal based on breath positions
    Arguments:
        signal{ndarray} -- The ECG signal  
        ann{ndarray} -- The ground truth R-Peaks
    Returns:
       ndarray -- transformed signal
    """
    
    length = len(signal)
    transform = []

    sample = 0
    if len(ann) == 0:
        return None
    if len(ann) ==1:
        for i in range(length):
            transform.append(abs(i-ann[sample]))
    else:
        for i in range(length):

            if sample+1 == len(ann):
                for j in range(i,length):

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


def getWindow(signal,ann, windows = 10, freq  = 125, overlap = 0.5):
    """Generate ECG and Respiration signals with annotations of specified window length
    
    Arguments:
        signal {2-D array} -- array containing ecg at index 0 and resp at index 1
        ann {list} -- annotations within specified window
    
    Keyword Arguments:
        windows {int} -- size of window in seconds (default: {5})
        freq {int} -- sampling rate in Hz (default: {125})
        overlap {float} -- percentage of overlap between windows (default: {0.5})
    
    Yields:
        tuple -- signals and correspoinding annotations
    """
    
    for start in range(0,len(signal),int((1-overlap)*freq*windows)):
        yield (signal[start: start + windows*freq, :],[x-start for x in ann if x >= start and x < start+windows*freq])
