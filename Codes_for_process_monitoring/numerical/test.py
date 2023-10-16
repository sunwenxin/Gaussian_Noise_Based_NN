import matplotlib.pyplot as plt
import numpy as np
from models import Normal_NN,G_NN
from scipy import stats

import matplotlib
from matplotlib import rcParams

#------------data generate-------------#
A = np.array([[1.8,0],[0,0.2]])
x_range = np.array([-1.5,1.5])
y_range = np.array([-1.5,1.5])
num = 100
xx, yy = np.meshgrid(  np.linspace(x_range[0], x_range[1], num,dtype='float32'),np.linspace(y_range[0], y_range[1], num,dtype='float32')  )
z = np.c_[xx.ravel(), yy.ravel()]
noise = stats.norm.rvs(size = z.shape)@A
y1 = np.sum(z**2,1,keepdims=True)
y2 = np.sin(z[:,0:1])+np.cos(z[:,1:2])
y = np.hstack((y1,y2)) + noise
#------------------------------------#


#------------testing-------------#
mod_our = G_NN([100]*3,['T'],lam=0,terms=5000,lr = 1e-3,alpha = 1e-2)
mod_our.train(z,y)
y_predict = mod_our.predict(z)
e_our = y - y_predict
m_e_our = np.mean(e_our,0)
st_e_our = (e_our-m_e_our).T@(e_our-m_e_our)/e_our.shape[0]




mod_Normal = Normal_NN([100]*3,['T'],lam=0,terms=5000,lr = 1e-2)
mod_Normal.train(z,y)
y_predict = mod_Normal.predict(z)
e_normal = y - y_predict
m_e_normal = np.mean(e_normal,0)
st_e_normal = (e_normal-m_e_normal).T@(e_normal-m_e_normal)/e_normal.shape[0]
#------------------------------------#
from scipy import io

#io.savemat('result.mat',{'sig':A@A,'e_our':e_our,'e_normal':e_normal})

rv = stats.multivariate_normal([0, 0], A@A)
rv_our = stats.multivariate_normal(m_e_our, st_e_our)
rv_normal = stats.multivariate_normal(m_e_normal, st_e_normal)

x_range = np.array([-3,3])*1.8
y_range = np.array([-3,3])*0.2
num = 500
xx, yy = np.meshgrid(  np.linspace(x_range[0], x_range[1], num,dtype='float32'),np.linspace(y_range[0], y_range[1], num,dtype='float32')  )
z = np.c_[xx.ravel(), yy.ravel()]

p = rv.pdf(z).reshape(xx.shape)
pmax = np.max(p)

p_our = rv_our.pdf(z).reshape(xx.shape)

p_normal = rv_normal.pdf(z).reshape(xx.shape)

plt.contour(xx,yy,p,np.linspace(0,pmax,7),alpha=0.75,colors='r')
plt.contour(xx,yy,p_our,np.linspace(0,pmax,7),alpha=0.75,colors='b')
plt.show()


plt.contour(xx,yy,p,np.linspace(0,pmax,7),alpha=0.75,colors='r')
plt.contour(xx,yy,p_normal,np.linspace(0,pmax,7),alpha=0.75,colors='b')
plt.show()
