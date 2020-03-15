import wfdb as wf
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
from scipy import signal
from sklearn.preprocessing import StandardScaler 
### Use standard scaler

def data_read(args):
    
    
    patient_no = args.patient_no
    patient_no -= 1

    path_dir = args.path_dir
    file_dirs = sorted(glob.glob(os.path.join(path_dir,'infant*')))   
    patient = {'ecg':[],'resp':[],'r_peaks':[],'resp_peaks':[],'brad_onset':[],'ecg_fs':[],'resp_fs':[]}
    
    for i in range(0,len(file_dirs),7):
        if (i == 7 * patient_no):
            
            print('-------- Acquiring data from patient {} --------'.format(patient_no + 1))

            ecg_file_name = os.path.splitext(file_dirs[i + 1])[0] 
            resp_file_name = os.path.splitext(file_dirs[i + 5])[0]
            qrsc_ext = os.path.splitext(file_dirs[i + 3])[1][1:]
            resp_ext = os.path.splitext(file_dirs[i + 6])[1][1:]
            brad_onset_ext = os.path.splitext(file_dirs[i + 6])[1][1:]
            atr_ext = os.path.splitext(file_dirs[i])[1][1:]
                
            ecg_sample_rate = wf.rdsamp(ecg_file_name)[-1]['fs']
            resp_sample_rate = wf.rdsamp(resp_file_name)[-1]['fs']
            ecg = wf.io.rdrecord(ecg_file_name).p_signal 
            resp = wf.io.rdrecord(resp_file_name).p_signal
            r_peaks_loc = wf.rdann(ecg_file_name,qrsc_ext).sample
            resp_peak_loc = wf.rdann(resp_file_name,resp_ext).sample
            brad_onset = wf.rdann(ecg_file_name,atr_ext).sample
                
            patient['ecg'].append(ecg)
            patient['resp'].append(resp)
            patient['r_peaks'].append(resp_peak_loc)
            patient['resp_peaks'].append(resp_peak_loc)
            patient['ecg_fs'].append(ecg_sample_rate)
            patient['resp_fs'].append(resp_sample_rate)
            patient['brad_onset'].append(brad_onset)

    return patient

def windowing_and_resampling_hr(patient):
    
    no_sec = 10
    final_ecg_sample_rate = 500
    final_resp_sample_rate = 50

    windowed_patient = {'ecg':[[] for i in range(10)],'resp':[[] for i in range(10)]}
    windowed_patient_overlap = {'ecg':[[] for i in range(10)],'resp':[[] for i in range(10)]}
    infant_no = 0

    for patients in patient[ 'ecg']:
        fs = patient['ecg_fs'][infant_no]
        window_len = no_sec * fs
        for i in range(len(patients) // (window_len-fs)):
            windowed_patient_overlap['ecg'][infant_no].append(patients[(window_len-fs) * i : ((window_len-fs) * i + window_len)])
            windowed_patient['ecg'][infant_no].append(patients[window_len * i : window_len * (i+1) ])
            if (final_ecg_sample_rate * no_sec) != window_len:
                windowed_patient_overlap['ecg'][infant_no][i] = signal.resample(windowed_patient_overlap['ecg'][infant_no][i],final_ecg_sample_rate * no_sec)
                windowed_patient['ecg'][infant_no][i] = signal.resample(windowed_patient['ecg'][infant_no][i],final_ecg_sample_rate * no_sec)
        
        infant_no += 1

    infant_no = 0
    for patients in patient['resp']:
        window_len = no_sec * patient['resp_fs'][infant_no] 
        for i in range(len(patients) // window_len):
            windowed_patient['resp'][infant_no].append(patients[window_len * i : window_len * (i+1)])
            if (final_resp_sample_rate * no_sec) != window_len:
                windowed_patient['resp'][infant_no][i] = custom_resample(windowed_patient['resp'][infant_no][i],patient['resp_fs'][infant_no])
        infant_no += 1

    return windowed_patient_overlap,windowed_patient

def windowing_and_resampling_br(patient):
    
    scaler = StandardScaler()
    no_sec = 10
    final_ecg_sample_rate = 500
    final_resp_sample_rate = 50

    windowed_patient = {'ecg':[[] for i in range(10)],'resp':[[] for i in range(10)]}
    windowed_patient_overlap = {'ecg':[[] for i in range(10)],'resp':[[] for i in range(10)]}
    infant_no = 0
    overlap_percent = 50
    overlap = int(50 / 100 * 5000)
    
    for patients in patient[ 'ecg']:
        
        fs = patient['ecg_fs'][infant_no]
        window_len = no_sec * fs
        for i in range(len(patients) // (window_len-overlap)):
            windowed_patient_overlap['ecg'][infant_no].append(scaler.fit_transform(patients[(window_len-overlap) * i : ((window_len-overlap) * i + window_len)]))
            if (final_ecg_sample_rate * no_sec) != window_len:
                windowed_patient_overlap['ecg'][infant_no][i] = signal.resample(windowed_patient_overlap['ecg'][infant_no][i],final_ecg_sample_rate * no_sec)
        
        infant_no += 1
    return windowed_patient_overlap

def custom_resample(resp,fs):
    
    modified_resp = []
    for i in range(int(len(resp) * 50/fs)):
        modified_resp.append(resp[int(fs/50*i)].astype(float)) 
    return np.asarray(modified_resp)
