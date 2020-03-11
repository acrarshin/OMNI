import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
import pandas as pd
import scipy.signal
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def testDataEval(model, loader, criterion):
    model.eval()
    
    with torch.no_grad():
        total_loss = 0
        for step,(x,y) in loader:
            ecg,BR = x.cuda(),y.cuda()
            BR_pred = model(ecg)
            
            loss = criterion(BR_pred, BR)
            total_loss+= loss
            break
        print (total_loss)

def smooth(signal,window_len=50):
    y = pd.DataFrame(signal).rolling(window_len,center = True, min_periods = 1).mean().values.reshape((-1,))
    return y

def findValleys(signal, prominence = 10, is_smooth = True , distance = 10):
    """ Return prominent peaks and valleys based on scipy's find_peaks function """
    smoothened = smooth(-1*signal)
    valley_loc = scipy.signal.find_peaks(smoothened, prominence= 0.07)[0]
    return valley_loc

def getBR(signal, model):
    model.eval()
    with torch.no_grad():
        transformPredicted = model(signal)
    transformPredicted = transformPredicted.cpu().numpy().reshape((-1,))
    valleys = findValleys(transformPredicted)
    return valleys, transformPredicted

def dist_transform(signal, ann):
    length = len(signal)
    transform = []

    sample = 0
    if len(ann) == 0:
        return None
    if len(ann) ==1:
        for i in range(len(signal)):
            transform.append(abs(i-ann[sample]))
    else:
        for i in range(len(signal)):

            if sample+1 == len(ann):
                for j in range(i,len(signal)):

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


def getWindow(signal,ann, windows = 5, freq = fs, overlap = 0.5):
    for start in range(0,len(signal),int((1-overlap)*freq*windows)):
        yield (signal[start: start + windows*freq, :],[x-start for x in ann if x >= start and x < start+windows*freq])
