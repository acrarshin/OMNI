from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.signal import resample
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import peak_finder,compute_heart_rate
from network import IncUNet_HR,IncUNet_BR

def get_input(window_size,input_dim):
    """ The input ECG is streamed here """
    """The input has to be streamed very cleverly"""
    
    streamed_input = np.random.random((5000))  #Write the Open BCI Streaming code here
    print(streamed_input.shape)
    return streamed_input
def main():
    
    actual_fs = 200
    window_size = 10
    C,H,W = 1,1,actual_fs*window_size
    required_fs = 500
    scaler = StandardScaler()
     
    loaded_model_HR = IncUNet_HR(in_shape=(C,H,W))    
    loaded_model_HR.load_state_dict(torch.load('../best_model_edt.pt', map_location = lambda storage, loc: storage)) ### Do change the paths
    loaded_model_HR.to('cpu')
    loaded_model_HR.eval()

    loaded_model_BR = IncUNet_BR(in_shape=(C,H,W))    
    loaded_model_BR.load_state_dict(torch.load('../best_model_bdt.pt', map_location = lambda storage, loc: storage)['model']) ### Do change the paths
    loaded_model_BR.to('cpu')
    loaded_model_BR.eval()
    
    r_peaks = []
    streaming = True
    plt.Figure()
    while(streaming):
        
        input = torch.from_numpy(scaler.fit_transform(resample(get_input(window_size,(C,H,W)).reshape(-1,1),required_fs * window_size))).unsqueeze(0).float().view(-1,1,5000)
        y_predict = loaded_model_HR(input)[0,:].detach().numpy()
        predicted_peak_locs = peak_finder(y_predict,input) 
        heart_rate = compute_heart_rate(np.asarray(predicted_peak_locs))
        input  = input[0,0,:].numpy()      
        
        plt.plot(input)
        plt.scatter(np.array(predicted_peak_locs),input[tuple(predicted_peak_locs)])
        plt.show()
        print('..........................')

        break

if __name__ == "__main__":
    main()