#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pylsl import StreamInlet, resolve_stream
import numpy as np
# %matplotlib notebook
from matplotlib import pyplot as plt
from network import IncUNet
import torch
#from IPython.display import clear_output


# In[ ]:


print("looking for an ECG stream...")
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])
#print(inlet)

C,H,W = 1,1,5000
loaded_model = IncUNet(in_shape=(C,H,W))    
loaded_model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location = lambda storage, loc: storage, pickle_module=pickle))
loaded_model.to(device)
loaded_model.eval()


sample_count =0
ecg_2s = []
while True:
    # get a new sample (you can also omit the timestamp part if you're not
    # interested in it)
    sample, timestamp = inlet.pull_sample()
    ecg_2s.append(sample) 
    sample_count+=1
    if(len(ecg_2s)==500):
        plt.close()
#        clear_output()
        ecg_2s = np.array(ecg_2s)*1e-6
        print(ecg_2s.shape)
        plt.plot(ecg_2s[:,0])
#        plt.show()
        plt.pause(0.5)
        plt.close()
        ecg_2s = []
#         
#         plt.plot(ecg_2s)
#         ecg_2s = []
#     print(sample[0])


# In[ ]:




