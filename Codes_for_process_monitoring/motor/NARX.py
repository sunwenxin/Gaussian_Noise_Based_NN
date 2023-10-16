from my_toolbox.dynamic_models.basic_dynamic import stacking
from my_toolbox.models.linear import whiten
import numpy as np

class NARX:
    def __init__(self,mod,order,Filter = True,whiten=True):
        self.mod = mod
        self.order = order
        self.Filter = Filter
        self.whiten = whiten
    def train(self,u,y):
        u = np.c_[u]
        y = np.c_[y]
        r = stacking(order = self.order,x = u,y = y,Filter = self.Filter)
        time_index = r['time_index']
        y_ = np.c_[r['y']]
        d = np.c_[r['vx'],r['vy']]
        if self.whiten:
            self.whiten = whiten(d)
            d = self.whiten.output(d)
        self.mod.train(d,y_)
    def predict(self,u,y):
        u = np.c_[u]
        y = np.c_[y]
        r = stacking(order = self.order,x = u,y = y,Filter = self.Filter)
        time_index = r['time_index']
        y_ = np.c_[r['y']]
        d = np.c_[r['vx'],r['vy']]
        if self.whiten:
            d = self.whiten.output(d)

        y_predict = np.zeros(y.shape)*np.nan
        y_predict[time_index,:] = self.mod.predict(d)
        return y_predict
        
