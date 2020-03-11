import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
from BRnet import IncUNet
from tqdm import tqdm_notebook as tqdm_nb
import tqdm as tqdm
from utils import testDataEval
from torch.utils.tensorboard import SummaryWriter

PATH = 'data'
X_train = torch.load(PATH + '/ecgtoBR_train_data.pt')
y_train = torch.load(PATH + '/ecgtoBR_train_labels.pt')

X_test = torch.load(PATH + '/ecgtoBR_test_data.pt')
y_test = torch.load(PATH + '/ecgtoBR_test_labels.pt')
BATCH_SIZE= 64
NUM_EPOCHS = 400
best_loss = 1000

train = TensorDataset(X_train,y_train)
val = TensorDataset(X_test,y_test)
trainLoader = DataLoader(train,batch_size = BATCH_SIZE,shuffle = True)
valLoader = DataLoader(val, batch_size= BATCH_SIZE, shuffle=True)

model = IncUNet((1,1,5000))
model.cuda()
criterion = torch.nn.SmoothL1Loss()
optim = torch.optim.Adam(model.parameters(),lr = 0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,milestones=[100,200], gamma=0.1)


NUM_EPOCHS = 400
best_loss = 1000



writer = SummaryWriter()

with tqdm(total = NUM_EPOCHS, desc = "Epochs") as pbar:
    for epoch in range(NUM_EPOCHS):
        
        print ("EPOCH: {}".format(epoch + 1), "Learning Rate: ", scheduler.get_lr()[0])
        model.train()
        totalLoss = 0
        for step,(x,y) in enumerate(trainLoader):
            print (".",end= "")
            ecg= x.cuda()
            BR = y.cuda()
            BR_pred = model(ecg)
            optim.zero_grad()
            loss = criterion(BR_pred,BR)
            totalLoss+=loss.cpu().item()
            loss.backward()
            optim.step()
        print ('')
        print ("Train Loss: ",totalLoss/(step+1))
        totalTestLoss = testDataEval(model, valLoader)
        scheduler.step()
        if best_loss > totalTestLoss:
            print ("Saving Best Model:")
            best_loss = totalTestLoss
            save_model("Best Model", epoch, model, optim, best_loss )
        
        writer.add_scalar("Loss/test",totalTestLoss, epoch )
        writer.add_scalar("Loss/train",totalLoss/(step+1),epoch )
        pbar.update()
        
writer.close()