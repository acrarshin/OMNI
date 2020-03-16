import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
from BRnet import IncUNet
from tqdm import tqdm_notebook as tqdm
import pandas as pd
import numpy as np
import scipy
import scipy.signal
import wfdb as wf
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import pickle
from torch.utils.data import Dataset
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

PATH = "../"
patient_nos = 10

class PreTermData(Dataset):
    def __init__(self,path = "path to preterm data", patient_nos = 10):
        super(PreTermData, self).__init__()
        self.ecg = []
        self.transform = []
        self.length_dict ={}
        
        for i in range(patient_nos):
            patient = str(i)
            fname_ecg = path + patient + "_ecg.pickle"
            fname_resp = path + patient + "_transform.pickle"

            with open(fname_ecg, "rb") as f:
                ecg = pickle.load(f)
            with open(fname_resp, "rb") as f:
                resp = pickle.load(f)
            assert len(ecg) == len(resp)
            self.ecg+=[torch.Tensor(i).type(torch.float) for i in ecg]
            self.transform+=[torch.Tensor(i).type(torch.float) for i in resp]
            self.length_dict[patient] = len(ecg)

def testDataEval(model, loader):
    model.eval()
    
    with torch.no_grad():
        total_loss = 0
        for step,(x,y) in enumerate(loader):
            ecg,BR = x.cuda(),y.cuda()
            BR_pred = model(ecg)
            
            loss = criterion(BR_pred, BR)
            total_loss+= loss.item()

        print ("Test Loss: ", total_loss/(step+1))
        
        return  total_loss/(step+1)

def save_model(exp_dir, epoch, model, optimizer,best_dev_loss):

    if not(os.path.exists(exp_dir)):
        os.mkdir(exp_dir)
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

    
def __getitem__(self,x):
        ecg = self.ecg[x]
        transform = self.transform[x]
        
        return ecg, transform
    
    def __len__(self):
        return len(self.ecg)

data = PreTermData()

torch.cuda.empty_cache()

loadState = torch.load("Best Model Preterm/best_model.pt")["model"]
loadedModel = IncUNet((1,1,5000))
loadedModel.load_state_dict(loadState)
loadedModel.cuda()
loadedModel.train()

for param in loadedModel.named_parameters():
    if(param[0][0] == 'e' and int(param[0][1]) < 6):
        param[1].requires_grad = False
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data[:][0], data[:][1], test_size = 0.2, random_state = 43)

x_train = torch.stack([i.cuda() for i in x_train])
x_test = torch.stack([i.cuda() for i in x_test])
y_train = torch.stack([i.cuda() for i in y_train])
y_test = torch.stack([i.cuda() for i in y_test])

trainData = TensorDataset(x_train,y_train)
trainLoader = DataLoader(trainData, batch_size = 64, shuffle = True)

testData = TensorDataset(x_test, y_test)
testLoader = DataLoader(testData, batch_size=64, shuffle = False)
# Train 

criterion = torch.nn.SmoothL1Loss()
optim = torch.optim.Adam(loadedModel.parameters(),lr = 0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,milestones=[100,200], gamma=0.1)

NUM_EPOCHS = 400
best_loss = 1000


if not(os.path.exists("transfer_runs")):
    os.mkdir("transfer_runs")
    

writer = SummaryWriter(log_dir ="transfer_runs")

with tqdm(total = NUM_EPOCHS, desc = "Epochs") as pbar:
    for epoch in range(NUM_EPOCHS):
        
        print ("EPOCH: {}".format(epoch + 1), "Learning Rate: ", scheduler.get_lr()[0])
        loadedModel.train()
        totalLoss = 0
        for step,(x,y) in enumerate(trainLoader):
            print (".",end= "")
            ecg= x.cuda()
            BR = y.cuda()
            BR_pred = loadedModel(ecg)
            optim.zero_grad()
            loss = criterion(BR_pred,BR)
            totalLoss+=loss.cpu().item()
            loss.backward()
            optim.step()
        print ('')
        print ("Train Loss: ",totalLoss/(step+1))
        totalTestLoss = testDataEval(loadedModel, testLoader)
        scheduler.step()
        if best_loss > totalTestLoss:
            print ("Saving Best Model:")
            best_loss = totalTestLoss
            save_model("Best Model Preterm", epoch, loadedModel, optim, best_loss )
        
        writer.add_scalar("Loss/test",totalTestLoss, epoch )
        writer.add_scalar("Loss/train",totalLoss/(step+1),epoch )
        pbar.update()
        
writer.close()