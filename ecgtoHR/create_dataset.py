import os
import glob
import wfdb as wf
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from scipy.signal import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch

from utils import dist_transform,getWindow

def data_preprocess(args):
    
    PATH = args.data_path
    all_paths = sorted(glob.glob(PATH + '/*.dat'))
    all_paths = [paths[:-4] for paths in all_paths]

    windowed_data,windowed_beats = getWindow(all_paths)

    scaler = StandardScaler()
    mod_windowed_data = []
    dist_tran_data = []

    for window in range(len(windowed_data)):
        
        beats = ((windowed_beats[window] * 500/360).astype(int))
        if(len(beats) != 0):
            mod_windowed_data.append(scaler.fit_transform(resample(windowed_data[window],5000).reshape(-1,1)))
            dist_tran_data.append(dist_transform(5000,beats))

    X_train,X_test,y_train,y_test = train_test_split(np.array(mod_windowed_data)[:,:,0],np.array(dist_tran_data)[:,:,0],test_size = 0.1, random_state = 42)

    X_train_toTensor = torch.Tensor(X_train).type(torch.float)
    print(X_train_toTensor.size())
    X_test_toTensor = torch.Tensor(X_test).type(torch.float)
    print(X_test_toTensor.size())
    y_train_toTensor = torch.Tensor(y_train).type(torch.float)
    print(y_train_toTensor.size())
    y_test_toTensor = torch.Tensor(y_test).type(torch.float)
    print(y_test_toTensor.size())

    if not(os.path.exists('data')):
        os.mkdir('data')

    torch.save(X_train_toTensor, "data/ecgtoHR_train_data.pt")
    torch.save(y_train_toTensor, "data/ecgtoHR_train_labels.pt")
    torch.save(X_test_toTensor, "data/ecgtoHR_test_data.pt")
    torch.save(y_test_toTensor, "data/ecgtoHR_test_labels.pt")

