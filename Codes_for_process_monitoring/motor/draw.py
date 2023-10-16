from scipy import io
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import rcParams
import numpy as np


matplotlib.use("svg")
pgf_config = {
    "font.family":'serif',
    "font.size": 10,
    "pgf.rcfonts": False,
    "text.usetex": True,
    "pgf.preamble": [
        r"\usepackage{unicode-math}",
        #r"\setmathfont{XITS Math}", 
        # 这里注释掉了公式的XITS字体，可以自行修改
        r"\setmainfont{Times New Roman}",
        r"\usepackage{xeCJK}",
        r"\xeCJKsetup{CJKmath=true}",
        r"\setCJKmainfont{SimSun}",
    ],
}
rcParams.update(pgf_config)


data = io.loadmat('result.mat')
t2_ls = data['t2_ls']
t2_our = data['t2_our']


bar_ls = data['bar_ls']
bar_our = data['bar_our']


fig, ax = plt.subplots(figsize=[6, 3])
ax.set_ylim(0, np.quantile(t2_ls[0,16::],0.995))
ax.set_xlim(0, t2_ls.size)
ax.grid(ls='--')
plt.subplots_adjust(top=0.9,bottom=0.15,left=0.1,right=0.97,hspace=0,wspace=0)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.plot(t2_ls[0,:],'b',lw=1.0)
ax.plot([0,t2_ls.size],[bar_ls[0,0],bar_ls[0,0]],'r', linestyle = 'dashed')
ax.set_xlabel('Time instant')
ax.set_ylabel('$T^2$')
plt.savefig("t2_ls.pdf")


fig, ax = plt.subplots(figsize=[6, 3])
ax.set_ylim(0, np.quantile(t2_our[0,16::],0.995))
ax.set_xlim(0, t2_our.size)
ax.grid(ls='--')
plt.subplots_adjust(top=0.9,bottom=0.15,left=0.1,right=0.97,hspace=0,wspace=0)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.plot(t2_our[0,:],'b',lw=1.0)
ax.plot([0,t2_our.size],[bar_our[0,0],bar_our[0,0]],'r', linestyle = 'dashed')
ax.set_xlabel('Time instant')
ax.set_ylabel('$T^2$')
plt.savefig("t2_our.pdf")
