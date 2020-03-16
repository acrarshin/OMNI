from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.signal import resample
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd

from utils import peak_finder,compute_heart_rate
from network import IncUNet_HR,IncUNet_BR

def get_input(window_size,input_dim):
    """ The input ECG is streamed here """
    """The input has to be streamed very cleverly"""
    
#     streamed_input = np.random.random((5000))  #Write the Open BCI Streaming code here
#     scatter(c = 'r')
    streamed_input = pd.read_csv('1_0.csv').values
    print(streamed_input.shape)
    return streamed_input
def main():

    actual_fs = 200
    window_size = 10
    C,H,W = 1,1,actual_fs*window_size
    required_fs = 500
    scaler = StandardScaler()

    loaded_model_HR = IncUNet_HR(in_shape=(C,H,W))    
    loaded_model_HR.load_state_dict(torch.load('../../best_model_edt.pt', map_location = lambda storage, loc: storage)) ### Do change the paths
    loaded_model_HR.to('cpu')
    loaded_model_HR.eval()

    loaded_model_BR = IncUNet_BR(in_shape=(C,H,W))    
    loaded_model_BR.load_state_dict(torch.load('../../best_model_bdt.pt', map_location = lambda storage, loc: storage)['model']) ### Do change the paths
    loaded_model_BR.to('cpu')
    loaded_model_BR.eval()
    
    r_peaks = []
    streaming = True
    count = 0
    while(streaming):
        incoming_input = get_input(window_size,(C,H,W)) 
        input = resample(incoming_input,5000).reshape(-1,1)    
        input = scaler.fit_transform(input,required_fs * window_size)
        input = torch.from_numpy(input).unsqueeze(0).float().view(-1,1,5000)
        y_predict = loaded_model_HR(input)[0,:].detach().numpy()
        predicted_peak_locs = peak_finder(y_predict,input) 
        heart_rate = compute_heart_rate(np.asarray(predicted_peak_locs))
        input  = input[0,0,:].numpy()      
        
        HR = 5
        BR = np.random.randint(12,16,1)
        
        print(BR.shape)
        
        plt.ion()   
        plt.figure(figsize=(25,25))
        
        plt.subplot(2,1,1)
        plt.plot(input)
        plt.title('HR:{} BR:{}'.format(HR,BR))
        # plt.legend(['signal'])

        plt.subplot(2,1,2)
        plt.plot(input)
        
        plt.scatter(np.array(predicted_peak_locs),input[tuple(predicted_peak_locs)], c = 'r')
        plt.legend(['signal','rpeak'])

        plt.show()
        plt.draw()
        plt.pause(5)
        plt.clf()
        
        print('..........................')
        count += 1
        if (count == 2):
                break
        

if __name__ == "__main__":
    main()