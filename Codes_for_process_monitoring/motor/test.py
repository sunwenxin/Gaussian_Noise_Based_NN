import matplotlib.pyplot as plt
import numpy as np
from models import Normal_NN,G_NN
from scipy import io
from my_toolbox.dynamic_models.basic_dynamic import stacking
from NARX import NARX
from scipy import io



data = io.loadmat('data.mat')
y = data['Y']
y_true = data['Y_true']
u = data['U']

order = 15

p_mod = Normal_NN([100]*1,['T'],terms=1000,lr=1e-3,lam=0)
mod_ls = NARX(p_mod,[order,order])
mod_ls.train(u,y)
y_predict = mod_ls.predict(u,y)
e_ls = y_predict-y
m_ls = np.nanmean(e_ls,0)
sig_ls = (e_ls[order+1:,:]-m_ls).T@(e_ls[order+1:,:]-m_ls)/(e_ls.shape[0]-order-1)

t2 = np.sum( np.linalg.solve(sig_ls,e_ls.T).T * e_ls ,1)
t2 = t2[order+1::]
t2.sort()
bar_ls = t2[int((t2.size*99.5)//100)]



p_mod = G_NN([100]*1,['T'],terms=1000,lr=1e-3,lam=0,alpha = 1e-8)
mod_our = NARX(p_mod,[order,0])
mod_our.train(u,y)
y_predict = mod_our.predict(u,y)
e_our = y_predict-y
m_our = np.nanmean(e_our,0)
sig_our = (e_our[order+1:,:]-m_our).T@(e_our[order+1:,:]-m_our)/(e_our.shape[0]-order-1)

t2 = np.sum( np.linalg.solve(sig_our,e_our.T).T * e_our ,1)
t2 = t2[order+1::]
t2.sort()
bar_our = t2[int((t2.size*99.5)//100)]


io.savemat('predict_error.mat',{'e_ls':e_ls[order+1::],'e_our':e_our[order+1::]})


data = io.loadmat('data_fault.mat')
y = data['Y_faulty']
y_true = data['Y_true']
u = data['U_faulty']



y_predict_ls = mod_ls.predict(u,y)
e_ls = (y_predict_ls-y)
spe_ls = np.sum(e_ls**2,1)
t2_ls = np.sum( np.linalg.solve(sig_ls,e_ls.T).T * e_ls ,1)

y_predict_our = mod_our.predict(u,y)
e_our = (y_predict_our-y)
spe_our = np.sum(e_our**2,1)
t2_our = np.sum( np.linalg.solve(sig_our,e_our.T).T * e_our ,1)


#io.savemat('result.mat',{'bar_ls':bar_ls,'bar_our':bar_our,'t2_ls':t2_ls,'t2_our':t2_our})

plt.figure(1)
plt.plot(t2_ls)
plt.plot([0,t2_ls.size],[bar_ls,bar_ls])

plt.figure(2)
plt.plot(t2_our)
plt.plot([0,t2_our.size],[bar_our,bar_our])
plt.show()
