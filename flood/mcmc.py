# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 10:21:17 2021

@author: ATANIM
"""

import numpy as np
import pymc3 as pm

observed_values = np.random.randn(100) # Here you would put your actual observations
with pm.Model() as model:
    obs_mu = pm.Normal('obs_mu', mu=1.3, sigma=0.05)
    obs_sigma = pm.Uniform('obs_sigma', lower=0, upper=1)
    obs = pm.Normal('obs', mu=obs_mu, sigma=obs_sigma,
                    observed=observed_values)
#%%
class Sub():
    pass
    
    def __init__(self,length, width):
        self.length = length
        self.width = width
    
    def area(self):
        return (self.length*self.width)
    def perim(self):
        return (self.length*2+ self.width*2)
    

l = [2,3,4,12]
w = [5,8,4,14]
for i,j in zip(l,w):
    rec = Sub(i,j)
    print(rec.area(), rec.perim())
#%%
#create
