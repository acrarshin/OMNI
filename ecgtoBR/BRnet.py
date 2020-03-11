import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class IncResBlock(nn.Module):
    def __init__(self, inplanes, planes, convstr=1, convsize = 15, convpadding = 7):
        super(IncResBlock, self).__init__()
        self.Inputconv1x1 = nn.Conv1d(inplanes, planes, kernel_size=1, stride = 1, bias=False)
        self.conv1_1 = nn.Sequential(
            nn.Conv1d(in_channels = inplanes,out_channels = planes//4,kernel_size = convsize,stride = convstr,padding = convpadding),
            nn.BatchNorm1d(planes//4))
        self.conv1_2 = nn.Sequential(
            nn.Conv1d(inplanes, planes//4, kernel_size=1, stride = convstr, padding=0, bias=False),
            nn.BatchNorm1d(planes//4),
            nn.LeakyReLU(0.2,),
            nn.Conv1d(in_channels = planes//4,out_channels = planes//4,kernel_size = convsize+2,stride = convstr,padding = convpadding+1),
            nn.BatchNorm1d(planes//4))
        self.conv1_3 = nn.Sequential(
            nn.Conv1d(inplanes, planes//4, kernel_size=1, stride = convstr, padding=0, bias=False),
            nn.BatchNorm1d(planes//4),
            nn.LeakyReLU(0.2,),
            nn.Conv1d(in_channels = planes//4,out_channels = planes//4,kernel_size = convsize+4,stride = convstr,padding = convpadding+2),
            nn.BatchNorm1d(planes//4))
        self.conv1_4 = nn.Sequential(
            nn.Conv1d(inplanes, planes//4, kernel_size=1, stride = convstr, padding=0, bias=False),
            nn.BatchNorm1d(planes//4),
            nn.LeakyReLU(0.2,),
            nn.Conv1d(in_channels = planes//4,out_channels = planes//4,kernel_size = convsize+6,stride = convstr,padding = convpadding+3),
            nn.BatchNorm1d(planes//4))
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = self.Inputconv1x1(x)

        c1 = self.conv1_1(x)
        c2 = self.conv1_2(x)
        c3 = self.conv1_3(x)
        c4 = self.conv1_4(x)

        out = torch.cat([c1,c2,c3,c4],1)
        out += residual
        out = self.relu(out)

        return out

class IncUNet (nn.Module):
    def __init__(self, in_shape):
        super(IncUNet, self).__init__()
        in_channels, height, width = in_shape
        self.e1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2,),
            IncResBlock(64,64))
        self.e2 = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv1d(64, 128, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm1d(128),
            IncResBlock(128,128))
        self.e2add = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(128))
        self.e3 = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2,),
            nn.Conv1d(128,256, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm1d(256),
            IncResBlock(256,256))
        self.e4 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv1d(256,256, kernel_size=4 , stride=1 , padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv1d(256,512, kernel_size=4, stride=2,padding=2),
            nn.BatchNorm1d(512),
            IncResBlock(512,512))
        self.e4add = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv1d(512,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(512)) 
        self.e5 = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv1d(512,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2,),
            nn.Conv1d(512,512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm1d(512),
            IncResBlock(512,512))

        self.e6 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv1d(512,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv1d(512,512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm1d(512), 
            IncResBlock(512,512))
        
        self.e6add = nn.Sequential(
            nn.Conv1d(512,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(512)) 
        
        self.e7 = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv1d(512,512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2,),
            nn.Conv1d(512,512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm1d(512),
            IncResBlock(512,512))
        
        self.e8 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.Conv1d(512,512, kernel_size=4, stride=1,padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv1d(512,512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm1d(512))
        
        
        self.d1 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.ConvTranspose1d(512, 512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2,),
            nn.ConvTranspose1d(512, 512, kernel_size=4, stride=1,padding =1),
            nn.BatchNorm1d(512),
            IncResBlock(512,512))
        
        self.d2 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.ConvTranspose1d(1024, 512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2,),
            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(512),
            IncResBlock(512,512))
        
        self.d3 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.ConvTranspose1d(1024, 512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            IncResBlock(512,512))
        
        self.d4 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.ConvTranspose1d(1024, 512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2,),
            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(512),
            IncResBlock(512,512))
        
        self.d5 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.ConvTranspose1d(1024, 512, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2,),
            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(512),
            IncResBlock(512,512))
        
        self.d6 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.ConvTranspose1d(1024, 512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(512),
            IncResBlock(512,512))
        
        self.d7 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.ConvTranspose1d(1024, 256, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2,),
            nn.ConvTranspose1d(256, 256, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(256),
            IncResBlock(256,256))
        
        self.d8 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.ConvTranspose1d(512, 128, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2,),
            nn.ConvTranspose1d(128, 128, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(128))
        
        self.d9 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(128))
        
        self.d10 = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.ConvTranspose1d(256, 64, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(64))
        
        self.out_l = nn.Sequential(
            nn.LeakyReLU(0.2,),
            nn.ConvTranspose1d(256, in_channels, kernel_size=3, stride=1,padding=1))
    
    
    def forward(self, x):       
        en1 = self.e1(x)
        #print (en1.size())
        en2 = self.e2(en1)
        #print (en2.size())
        en2add = self.e2add(en2)
        #print (en2add.size())
        en3 = self.e3(en2add)
        #print (en3.size())
        en4 = self.e4(en3)
        #print (en4.size())
        en4add = self.e4add(en4)
        #print (en4add.size())
        en5 = self.e5(en4add)
        #print (en5.size())
        en6 = self.e6(en5)
        #print (en6.size())
        en6add = self.e6add(en6)
        #print (en6add.size())
        en7 = self.e7(en6add)
        #print (en7.size())
        en8 = self.e8(en7)
        #print (en8.size())
        
        de1_ = self.d1(en8)
        #print (de1_.size())
        de1 = torch.cat([en7,de1_],1)
        #print (de1.size())
        de2_ = self.d2(de1)
        #print (de2_.size())
        #de2_ = nn.ConstantPad1d((0,1),0)(de2_)
        de2 = torch.cat([en6add,de2_],1)
        #print (de2.size())
        de3_ = self.d3(de2)
        #print (de3_.size())
        de3 = torch.cat([en6,de3_],1)
        #print (de3.size())
        de4_ = self.d4(de3)
        #print (de4_.size())
        #de4_ = nn.ConstantPad1d((0,1),0)(de4_)
        de4 = torch.cat([en5,de4_],1)
        #print (de4.size())
        de5_ = self.d5(de4)
        #print (de5_.size())
        de5_ = nn.ConstantPad1d((0,1),0)(de5_)
        de5 = torch.cat([en4add,de5_],1)
        #print (de5.size())
        de6_ = self.d6(de5)
        #print (de6_.size())
        de6 = torch.cat([en4,de6_],1)
        #print (de6.size())
        de7_ = self.d7(de6)
        #print ("here",de7_.size())
        de7_ = de7_[:,:,:-1]
        de7 = torch.cat([en3,de7_],1)
        #print (de7.size())
        de8 = self.d8(de7)
        #print (de8.size())
        de8_ = self.d8(de7)
        #print (de8_.size())
        de8 = torch.cat([en2add,de8_],1)
        #print (de8.size())
        out = self.out_l(de8)
        return out
    