import numpy as np
import matplotlib.pyplot as plt
from scipy import stats,io


from sklearn.neighbors import KernelDensity

err = io.loadmat('predict_error.mat')

e_normal = err['e_ls']

e_our = err['e_our']


data = io.loadmat('data.mat')
y = data['Y']
y_true = data['Y_true']
e = y-y_true

sig = e.T@e/e.shape[0]


bandwidth = 0.2
rang = 2.6

def scatter_hist(data, ax, ax_histx, ax_histy):
    x_range = np.array([-1,1])*np.sqrt(sig[0,0])*rang
    y_range = np.array([-1,1])*np.sqrt(sig[1,1])*rang
    num = 500
    xx, yy = np.meshgrid(  np.linspace(x_range[0], x_range[1], num,dtype='float32'),np.linspace(y_range[0], y_range[1], num,dtype='float32')  )
    z = np.c_[xx.ravel(), yy.ravel()]
    ax.grid(ls='--')
    ax_histx.grid(ls='--')
    ax_histy.grid(ls='--')
    rv = stats.multivariate_normal([0, 0], sig)

    dens = np.exp( KernelDensity(kernel="linear", bandwidth=bandwidth).fit(data).score_samples(z) )
    dens = dens.reshape(xx.shape)
    p = rv.pdf(z).reshape(xx.shape)

    strs = np.linspace(-1e-8,np.max(p)-0.005,6)
    strs = ((strs*1000 )//1)/1000
    
    CS1 = ax.contour(xx,yy,p,strs,alpha=1.0,colors='r')
    CS2 = ax.contour(xx,yy,dens,strs,alpha=1.0,colors='b')
    
    h1,_ = CS1.legend_elements()
    h2,_ = CS2.legend_elements()
    ax.legend([h1[0], h2[0]], ['PDF of $\mathbf d$', 'PDF of $\mathbf y^{\mathbf d}-\hat\mathbf y^{\mathbf d}$'])
    
    fmt1 = {}
    fmt2 = {} 
    for l1, l2, s in zip(CS1.levels,CS2.levels, strs):
        fmt1[l1] = s
        fmt2[l2] = s
    plt.clabel(CS1, CS1.levels, inline = True, 
               fmt = fmt1, fontsize = 10)
    plt.clabel(CS2, CS2.levels, inline = True, 
           fmt = fmt2, fontsize = 10) 
    ax.set_xlim(x_range[0],x_range[1])
    ax.set_ylim(y_range[0],y_range[1])
    ax_histx.spines['right'].set_color('none')
    ax_histx.spines['top'].set_color('none')

    ax_histy.spines['bottom'].set_color('none')
    ax_histy.spines['right'].set_color('none')
    ax_histy.xaxis.set_ticks_position('top')
    
    rv = stats.multivariate_normal([0,], sig[0,0])
    z = np.linspace( x_range[0],x_range[1], num )
    dens = np.exp( KernelDensity(kernel="linear", bandwidth=bandwidth).fit(data[:,0:1]).score_samples(np.c_[z]) )
    p = rv.pdf(z)
    ax_histx.plot(z,p,'r')
    ax_histx.fill_between(z, p, 0, facecolor='r', alpha=0.6)
    ax_histx.plot(z,dens,'b')
    ax_histx.fill_between(z, dens, 0, facecolor='b', alpha=0.6)
    #ax_histx.set_ylim(0,0.12)

    rv = stats.multivariate_normal([0,], sig[1,1])
    z = np.linspace( y_range[0],y_range[1], num )
    dens = np.exp( KernelDensity(kernel="linear", bandwidth=bandwidth).fit(data[:,1:2]).score_samples(np.c_[z]) )
    p = rv.pdf(z)
    ax_histy.plot(p,z,'r')
    ax_histy.plot(dens,z,'b')
    ax_histy.fill_between( np.r_[0,p,0], np.r_[z[0],z,z[-1]], facecolor='r', alpha=0.6)
    ax_histy.fill_between( np.r_[0,dens,0], np.r_[z[0],z,z[-1]], facecolor='b', alpha=0.6)
    #ax_histy.set_xlim(0,0.8)

    ax.set_xlabel('$\mathbf d_1$ and $\mathbf y^{\mathbf d}_1-\hat\mathbf y^{\mathbf d}_1$')
    ax.set_ylabel('$\mathbf d_2$ and $\mathbf y^{\mathbf d}_2-\hat\mathbf y^{\mathbf d}_2$')

    ax_histx.set_ylabel('PDF')
    ax_histy.set_xlabel('PDF')

    


from matplotlib import pyplot as plt
import matplotlib
from matplotlib import rcParams
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

    

fig = plt.figure(figsize=(6, 5))
gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.15, right=0.95, bottom=0.1, top=0.9,
                      wspace=0.08, hspace=0.08)

ax = fig.add_subplot(gs[1, 0])
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
ax_histx.tick_params(axis="x", labelbottom=False)
ax_histy.tick_params(axis="y", labelleft=False)
scatter_hist(e_our, ax, ax_histx, ax_histy)

plt.savefig("usetex_our.pdf")






fig = plt.figure(figsize=(6, 5))
gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.15, right=0.95, bottom=0.1, top=0.9,
                      wspace=0.08, hspace=0.08)

ax = fig.add_subplot(gs[1, 0])
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
ax_histx.tick_params(axis="x", labelbottom=False)
ax_histy.tick_params(axis="y", labelleft=False)
scatter_hist(e_normal, ax, ax_histx, ax_histy)

plt.savefig("usetex_ls.pdf")
