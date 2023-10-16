import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.nn import Linear as Li
import numpy as np

class Net(torch.nn.Module):
    def __init__(self,nods,a_f,input_dim,output_dim):
        super(Net, self).__init__()
        Act = {'LR':nn.LeakyReLU,'T':nn.Tanh,'R':nn.ReLU,'S':nn.Sigmoid}
        nods = nods
        a_f = a_f
        self.layers = len(nods)

        if len(a_f) != self.layers:
            if len(a_f)==1:
                a_f *= self.layers
            else:
                print('please check the deepth of your network')

        self.input = torch.nn.Sequential( Li(input_dim, nods[0]),Act[a_f[0]]() )
        self.f = nn.ModuleList( [torch.nn.Sequential( Li(nods[i], nods[i+1]),Act[a_f[i+1]]() ) for i in range(self.layers-1)] )
        self.output = Li(nods[-1], output_dim)

    def forward(self, x):
        x =  self.input(x)
        for i in range(self.layers-1):
            x = self.f[i](x)
        x = self.output(x)
        return x
         




#-------------------------------------------------------------------------------------------------------#


class Normal_NN:
    def __init__(self,nods,a_f,terms=2000,lam = 1e-3,lr = 1e-3):
        self.lr = lr
        self.lam = lam
        self.nods = nods
        self.a_f = a_f
        self.terms = terms
        self.initialized = False
    def train(self,x,y):
        x = np.c_[x]
        y = np.c_[y]
        if not self.initialized:
            [num,input_dim] = x.shape
            output_dim = y.shape[1]
            self.mod = Net(self.nods,self.a_f,input_dim,output_dim)
            self.optimizer = torch.optim.Adam(self.mod.parameters(), lr=self.lr, weight_decay=self.lam)
            self.initialized = True
        x = x.astype('float32')
        x = torch.from_numpy(x)
        y = y.astype('float32')
        y = torch.from_numpy(y)
        loss_func = torch.nn.MSELoss()
        for t in range(self.terms):
            prediction = self.mod(x)
            loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)
            self.optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            self.optimizer.step()        # apply gradients
            print('%d'%t+':%f'%loss)
    def predict(self,x):
        x = np.c_[x]
        x = x.astype('float32')
        x = torch.from_numpy(x)
        y = self.mod(x)
        return y.data.numpy()

#-------------------------------------------------------------------------------------------------------#

class G_NN:
    def __init__(self,nods,a_f,terms=2000,lam = 1e-3,lr = 1e-3,alpha = 1e-3):
        self.lr = lr
        self.lam = lam
        self.nods = nods
        self.a_f = a_f
        self.terms = terms
        self.initialized = False
        self.alpha = alpha
    def train(self,x,y):
        x = np.c_[x]
        y = np.c_[y]
        [num,input_dim] = x.shape
        if not self.initialized:
            output_dim = y.shape[1]
            self.mod = Net(self.nods,self.a_f,input_dim,output_dim)
            self.optimizer = torch.optim.Adam(self.mod.parameters(), lr=self.lr, weight_decay=self.lam)
            self.initialized = True
        x = x.astype('float32')
        x = torch.from_numpy(x)
        y = y.astype('float32')
        y = torch.from_numpy(y)
        loss_func = torch.nn.MSELoss()
        for t in range(self.terms):
            prediction = self.mod(x)
            e = prediction-y
            M = e.T@e/num
            a,b = torch.linalg.slogdet(M)
            loss = a*b + self.alpha*loss_func(prediction, y)     # must be (1. nn output, 2. target)
            self.optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            self.optimizer.step()        # apply gradients
            print('%d'%t+':%f'%loss)
    def predict(self,x):
        x = np.c_[x]
        x = x.astype('float32')
        x = torch.from_numpy(x)
        y = self.mod(x)
        return y.data.numpy()

#--------------------------------------------------------
