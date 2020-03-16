# OMNI (Open Source Monitoring of Neonates and Infants) 

<p align="center">
  <image src = 'images/omni-logo.png' >
</p>

## Software Requirements
Run `sh.requirements.sh` in a virtual environment in order to download the required libraries. 
## System Configuration
* Ubuntu 16.04
* Nvidia 1080Ti - (Required for training the model)

## Train a model to extract R peaks and Heart Rate from ECG waveform.  
* Download ECG MITDB monitoring data from https://storage.googleapis.com/mitdb-1.0.0.physionet.org/mit-bih-arrhythmia-database-1.0.0.zip and unzip it.
* To train: `python train.py --preprocess_data --data_path "PATH TO DATA"` 

## Train a model to extract Breathing Rate from ECG waveform. 
* Download ECG from the BIDMC database which is derived from MIMIC-II from https://physionet.org/static/published-projects/bidmc/bidmc-ppg-and-respiration-dataset-1.0.0.zip and unzip it. 
* To train: `python train.py --preprocess_data --data_path "PATH TO DATA"`

## Model Inference
* Download ECG from the preterm infant database from https://physionet.org/static/published-projects/picsdb/preterm-infant-cardio-respiratory-signals-database-1.0.0.zip and unzip it. 
* Download the Heart Rate computation model from here: https://drive.google.com/open?id=1yI7G4nofjuzFWkD1CfsOtLZxaukTu0di
* Download the Breathing Rate computation model from here: https://drive.google.com/open?id=1ycV74LfGmgcGmLlrPn2VileeFNsGrRZT
* To run inference and view GUI type: `python run_model.py --path_dir "PATH TO DATA" --saved_hr_model_path "PATH TO HR MODEL" --saved_br_model_path "PATH TO BR MODEL" --patient_no 3 --viewer 1`

# OMNI OpenBCI Pi Inference

 Edge inference of ECG R-peak detection and Respiration extraction using Raspberry Pi 4 using ECG (OpenBCI Ganglion).


## Hardware Design

Wearable ECG electrodes --> OpenBCI Ganglion ---(Bluetooth)---> Raspberry Pi 4

## Software Design

OpenBCI client ----(LSL)--->  Python -> PyTorch inference --> Breathing Rate, Heart Rate


## Installation instruction

### Install PyTorch on Raspberry Pi 4:

 1. Install PyTorch dependicies 
 `sudo apt install libopenblas-dev libblas-dev m4 cmake cython python3-yaml libatlas-base-dev`
 2. Increase swap file memory to 1600, Edit variable `CONF_SWAPSIZE` in `/etc/dphys-swapfile`
 3. Reset environmental variables like ONNX_ML [Instructions](https://gist.github.com/akaanirban/621e63237e63bb169126b537d7a1d979) 
 4. Download PyTorch package compiled for Armv7 ([torch-1.1.0-cp37-cp37m-linux_armv7l.whl](https://github.com/marcusvlc/pytorch-on-rpi/blob/master/torch-1.1.0-cp37-cp37m-linux_armv7l.whl))
 5. Install using the command `sudo pip3 install torch-1.1.0-cp37-cp37m-linux_armv7l.whl` in the same directory

Refer [here](https://github.com/marcusvlc/pytorch-on-rpi) for troubleshooting 

### Install OpenBCI Ganglion client on Raspberry Pi 4:

1. Clone OpenBCI_Python repo
 `git clone htps://github.com/OpenBCI/OpenBCI_Python.git`
2. Install the following requisites python packages using `pip3 install`
	pylsl, python-osc, six, socketIO-client, websocket-client, Yapsy, xmldict, bluepy
3. Open folder OpenBCI_Python and run   
    `sudo python3 user.py --board ganglion -a streamer_lsl` to open a lab streaming layer stream of sensor data from the ganglion
    
## Instructions to run to perform real time breathing rate/ heart rate inference using OpenBCI data
1. Run the lsl streamer script to get data to the inference script
`sudo python3 user.py --board ganglion -a streamer_lsl`
3. Run the visualization and edge inference code on the pi using  `python3 lsl_openbci.py`
