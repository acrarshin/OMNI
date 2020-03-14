# Smart Infant Monitoring

### Train a model to extract R peaks and Heart Rate from ECG.  
* Download ECG MITDB monitoring data from https://storage.googleapis.com/mitdb-1.0.0.physionet.org/mit-bih-arrhythmia-database-1.0.0.zip and unzip it.
* To train: python train.py --preprocess_data --data_path "PATH TO DATA"

### Train a model to extract Breathing waveform peaks and Heart Rate from Breathing Rate. 
* Download ECG from the BIDMC database which is derived from MIMIC-II from https://physionet.org/static/published-projects/bidmc/bidmc-ppg-and-respiration-dataset-1.0.0.zip and unzip it.
* To train: python train.py --preprocess_data --data_path "PATH TO DATA"

### Model Inference
* Download ECG from the preterm infant database from https://physionet.org/static/published-projects/picsdb/preterm-infant-cardio-respiratory-signals-database-1.0.0.zip and unzip it. 
* Download the R Peak computation model from here: https://drive.google.com/open?id=1yI7G4nofjuzFWkD1CfsOtLZxaukTu0di
* To run inference and view GUI type: python run_model.py --path_dir "PATH TO DATA" --saved_model_path "PATH TO MODEL" --patient_no 3 --viewer 1

