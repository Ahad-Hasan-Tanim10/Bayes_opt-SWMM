# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 10:10:56 2021

@author: ATANIM
"""

from swmm_api.input_file.section_labels import SUBCATCHMENTS,SUBAREAS, INFILTRATION, TRANSECTS
from swmm_api.input_file import read_inp_file, SwmmInput, section_labels as sections
import numpy as np
from pyswmm import Simulation, Subcatchments, Nodes, Links
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

inp = read_inp_file(r'C:\Users\ATANIM\Documents\Research\Rocky branch\Flood\Calibration_m.inp', convert_sections=[SUBCATCHMENTS,SUBAREAS, INFILTRATION, TRANSECTS])  # type: swmm_api.SwmmInput

#%%
class Subcatchment():
    
    def __init__(self, name):
        self.name = name
        self.value = None
        
    def setv(self, value):
        pass

#class sw(Subcatchment):
#    def __init__(self, width):
#        inp[sections.SUBCATCHMENTS][self.name].Width = setv(self.width)
#        return self.width
###        inp[sections.SUBCATCHMENTS][self].Width = self.width
        
#%%
subcatch1 = Subcatchment('ws-128')
subcatch1.value = 286.6

#%%...
def sub(x,a):
    inp[sections.SUBCATCHMENTS][x].Width = a 
    return a
#%%
def cn(x,a):
    inp[sections.INFILTRATION][x].Psi = a 
    return a
#%%
def sl(x,a):
    inp[sections.SUBCATCHMENTS][x].Slope = a 
    return a
#%%
def Nimp(x,a):
    inp[sections.SUBAREAS][x].N_Imperv = a
    return a
#%% step
def Nperv(x,a):
    inp[sections.SUBAREAS][x].N_Perv = a
    return a
#%%
def manch(x,a):
    inp[sections.TRANSECTS][x].roughness_channel = a
    return a
def manchr(x,a):
    inp[sections.TRANSECTS][x].roughness_right = a
    return a
def manchl(x,a):
    inp[sections.TRANSECTS][x].roughness_left = a
    return a
#%%
catchment = ['ws-128', 'ws-148', 'ws-159', 'ws-71', 'ws-84', 'ws-161', 'ws-17']  
sw = []
sN = []
Nimpl = []
Npervl = []
for i in catchment:
    rd = inp[sections.SUBCATCHMENTS][i].Width
    sw.append(rd)
    cd = inp[sections.INFILTRATION][i].Psi
    sN.append(cd)
    im = inp[sections.SUBAREAS][i].N_Imperv
    Nimpl.append(im)
    per = inp[sections.SUBAREAS][i].N_Perv
    Npervl.append(per)
    print(Npervl)
sw1 = []
sN1 = []
for i in sw:
    sa = 1.05*i
    sw1.append(sa)
for i in sN:
    ra = i+2
    sN1.append(ra)
#%% No roughness is updated
Cond = ['c67', 'c68', 'c70', 'c71']    
n_chan = []
n_left = []
n_right = []
for i in Cond:
    rc = inp[sections.TRANSECTS][i].roughness_channel
    n_chan.append(rc)
    rr = inp[sections.TRANSECTS][i].roughness_right
    n_right.append(rr)
    rl = inp[sections.TRANSECTS][i].roughness_left
    n_left.append(rl)
    print(n_chan)
#%%
vf_dis = pd.read_csv(r'C:/Users/ATANIM/Documents/Research/Rocky branch/Validation/02169505discharge.txt',  sep='\t')
vf_dis['Time'] =  pd.to_datetime(vf_dis['datetime'], format='%m/%d/%Y %H:%M')
vf_dis1 = vf_dis.set_index(['Time']).shift(1)
vf_dis1.pop('datetime')
vf_dis1 = vf_dis1.resample("10min").mean().fillna(0)
#%%
def dis(mod, obs):
    dep = mod.resample("10min").mean().to_frame()
    d1 = dep["S1 Discharge"].to_numpy()
    d2 = ((obs["discharge"]*0.027).to_numpy())#[:-2]
    n = d2.size - d1.size
    index = [i for i in range(0,n)]
    d3 = np.delete(d2, index)
    #r2_dis = ((np.corrcoef(d1, d3))[0,1])**2
    nse = r2_score(d3, d1)
    return nse
#mod = df[S1 Discharge], obs = vf_dis
#%%
def runf(fn):
    f_path = r'C:\Users\ATANIM\Documents\Research\Rocky branch\Flood'
    sim_path = os.path.join(f_path, "%s.inp"% fn)
    with Simulation(sim_path) as sim:
        S1 = Links(sim)["usgs_dis"]
        S2 = Nodes(sim)["43_depth"]   
    # we need to create a time series table, idx is the time, s1_values, s2_values are the two columns for the runoff of s1 and s1
        idx = []
        s1_values = []
        s2_values = []   
    # as we are looping through each time step, we add the simulated value into the 3 coloumns variables above
        for step in sim:
            idx.append(sim.current_time)
            s1_values.append(S1.flow)
            s2_values.append(S2.depth)   
    # using this line below, we turn the 3 columns into a table, called DataFrame using the pandas library. So that we can plot it.
        df = pd.DataFrame({'S1 Discharge': s1_values, 'S2 Depth': s2_values}, index=idx)
        return df
#%%
import os
def inpfi(cnu, Nimplr, Nperlv, n_chan, n_right, n_left, fn):
    for i,j in zip(catchment, cnu):
        cn(i,j)
    for e,f in zip(catchment, Nimplr):
        Nimp(e,f)
    for g,h in zip(catchment,Nperlv):
        Nperv(g,h)
    for k, l in zip(Cond, n_chan):
        manch(k,l)
    for a, b in zip(Cond, n_right):
        manchr(a, b)
    for c, d in zip(Cond, n_left):
        manchr(c, d)
        f_path = r'C:\Users\ATANIM\Documents\Research\Rocky branch\Flood'
        return inp.write_file(os.path.join(f_path, "%s.inp"% fn))
#%%
catchment = ['ws-128', 'ws-148', 'ws-159', 'ws-71', 'ws-84', 'ws-161', 'ws-17']  
sw = []
sN = []
for i in catchment:
    cd = inp[sections.INFILTRATION][i].Psi
    sN.append(cd)
    print(sN)
#%%
def modelswmmi(a,b,c,d,e,f,g,
               a1, b1, c1, d1, e1, f1, g1,
               a2, b2, c2, d2, e2, f2, g2,
               i,j,k,l, h, m, n,o, p, q, r, s, fn):
    CNU = [cn('ws-128',(sN[0]+a)), cn('ws-148',(sN[1]+b)),cn('ws-159',(sN[2]+c)), cn('ws-71',(sN[3]+d)), cn('ws-84',(sN[4]+e)),
           cn('ws-161',(sN[5]+f)), cn('ws-17',(sN[6]+g))]
    n_imp = [Nimp('ws-128',(Nimpl[0]+a1)), Nimp('ws-148',(Nimpl[1]+b1)),Nimp('ws-159',(Nimpl[2]+c1)), Nimp('ws-71',(Nimpl[3]+d1)), Nimp('ws-84',(Nimpl[4]+e1)),
           Nimp('ws-161',(Nimpl[5]+f1)), Nimp('ws-17',(Nimpl[6]+g1))]
    n_per = [Nperv('ws-128',(Npervl[0]+a2)), Nperv('ws-148',(Npervl[1]+b2)),Nperv('ws-159',(Npervl[2]+c2)), Nperv('ws-71',(Npervl[3]+d2)), Nperv('ws-84',(Npervl[4]+e2)),
           Nperv('ws-161',(Npervl[5]+f2)), Nperv('ws-17',(Npervl[6]+g2))]
    n_ch = [manch('c67',(n_chan[0]+i)),(manch('c68',n_chan[1])+j), (manch('c70',n_chan[2])+k), (manch('c71',n_chan[3])+l)]
    n_r = [manchr('c67',(n_right[0]+h)), manchr('c68',(n_right[1]+m)), manchr('c70',(n_right[2]+n)), manchr('c71',(n_right[3]+o))]
    n_l = [manchl('c67',(n_left[0]+p)), manchl('c68',(n_left[1]+q)), manchl('c70',(n_left[2]+r)), manchl('c71',(n_left[3]+s))]
    inpfi(CNU,n_imp, n_per, n_ch,n_r,n_l, fn)
    df = runf(fn)
    r2 = dis(df["S1 Discharge"], vf_dis1)
    return r2
#%%
def fr33(a, b, c, d, e, f, g,a1, b1, c1, d1, e1, f1, g1, a2, b2, c2, d2, e2, f2, g2,i,j,k,l, h, m, n,o,p, q, r, s):
    # print(params)  # <-- you'll see that params is a NumPy array
    # <-- for readability you may wish to assign names to the component variables
    r2 = modelswmmi(a, b, c, d,  e, f, g, a1, b1, c1, d1, e1, f1, g1,a2, b2, c2, d2, e2, f2, g2, i,j,k,l, h, m, n,o,p, q, r, s, "cal2")
    return r2
#%%
#def modelswmmi(a, fn):
#    sw7 = [cn(catchment,a)]
#    inpfi(sw7, fn)
#    df = runf(fn)
#    r2 = dis(df["S1 Discharge"], vf_dis)
#    return r2

#%%
a = np.linspace(-5, 10, 50)
for i in (a):
    p = []
    p = modelswmmi(i, "cal2")
    print(i,p)
#%%
from bayes_opt import BayesianOptimization
from bayes_opt import SequentialDomainReductionTransformer

#bounds_transformer = SequentialDomainReductionTransformer()

pbounds = {'a': (-5,10), 'b': (-5,10), 'c': (-5,10), 'd': (-5,10),
           'e': (-5,10), 'f': (-5,10), 'g': (-5,10) }

optimizer = BayesianOptimization(f = fr7, pbounds = pbounds, verbose =2,
     random_state =9, #bounds_transformer=bounds_transformer
 )

optimizer.maximize(init_points =20,
                   n_iter =30,
                   acq = 'ucb'
                   
        )
#%%
import sherpa
parameters = [sherpa.Continuous(name='a', range=[-5,10]),
              sherpa.Continuous(name='b', range=[-6,10]),
              sherpa.Continuous(name='c', range=[-6,10]),
              sherpa.Continuous(name='d', range=[-6,10]),
              sherpa.Continuous(name='e', range=[-6,10]),
              sherpa.Continuous(name='f', range=[-6,10]),
              sherpa.Continuous(name='g', range=[-6,10]),
              sherpa.Continuous(name='a1', range=[0.01,0.9]),
              sherpa.Continuous(name='b1', range=[0.01,0.9]),
              sherpa.Continuous(name='c1', range=[0.01,0.9]),
              sherpa.Continuous(name='d1', range=[0.01,0.9]),
              sherpa.Continuous(name='e1', range=[0.01,0.9]),
              sherpa.Continuous(name='f1', range=[0.01,0.7]),
              sherpa.Continuous(name='g1', range=[0.01,0.9]),
              sherpa.Continuous(name='a2', range=[-0.35,0.6]),
              sherpa.Continuous(name='b2', range=[-0.35,0.6]),
              sherpa.Continuous(name='c2', range=[-0.35,0.6]),
              sherpa.Continuous(name='d2', range=[-0.45,0.4]),
              sherpa.Continuous(name='e2', range=[-0.35,0.6]),
              sherpa.Continuous(name='f2', range=[-0.35,0.6]),
              sherpa.Continuous(name='g2', range=[-0.35,0.6]),
              sherpa.Continuous(name='i', range=[-0.005,0.035]),
              sherpa.Continuous(name='j', range=[0.005,0.035]),
              sherpa.Continuous(name='k', range=[-0.005,0.025]),
              sherpa.Continuous(name='l', range=[+0.005,0.035]),
              sherpa.Continuous(name='h', range=[-0.005,0.035]),
              sherpa.Continuous(name='m', range=[-0.001,0.095]),
              sherpa.Continuous(name='n', range=[-0.005,0.035]),
              sherpa.Continuous(name='o', range=[-0.005,0.055]),
              sherpa.Continuous(name='p', range=[-0.005,0.035]),
              sherpa.Continuous(name='q', range=[-0.005,0.035]),
              sherpa.Continuous(name='r', range=[-0.005,0.015]),
              sherpa.Continuous(name='s', range=[0.005,0.015])]
algorithm = sherpa.algorithms.GPyOpt(model_type='GP',
                                     acquisition_type='MPI', 
                                     max_concurrent=1,)
#algorithm = sherpa.algorithms.PopulationBasedTraining(population_size=5,
#                                                      num_generations=5,
#                                                      perturbation_factors=(0.8, 1.2))

study = sherpa.Study(parameters=parameters,
                     algorithm=algorithm,
                     lower_is_better=False,
                     disable_dashboard=True)
#result = open(r'C:\Users\ATANIM\Documents\Research\Rocky branch\Calibration\optimization.txt', 'w')
#result.close()
for trial in study:
    print("Trial {}:\t{}".format(trial.id, trial.parameters))

    num_iterations = 3
    for i in range(num_iterations):
        
        # access parameters via trial.parameters and id via trial.id
        pseudo_loss = fr33(trial.parameters['a'], trial.parameters['b'], 
                          trial.parameters['c'], trial.parameters['d'],
                          trial.parameters['e'],trial.parameters['f'],
                          trial.parameters['g'], trial.parameters['a1'],
                          trial.parameters['b1'], trial.parameters['c1'],
                          trial.parameters['d1'],trial.parameters['e1'],
                          trial.parameters['f1'],trial.parameters['g1'],
                           trial.parameters['a2'],
                          trial.parameters['b2'], trial.parameters['c2'],
                          trial.parameters['d2'],trial.parameters['e2'],
                          trial.parameters['f2'],trial.parameters['g2'],
                          trial.parameters['i'],
                          trial.parameters['j'], trial.parameters['k'],
                          trial.parameters['l'], trial.parameters['h'],
                          trial.parameters['m'],trial.parameters['n'],
                          trial.parameters['o'],trial.parameters['p'],
                          trial.parameters['q'],trial.parameters['r'],
                          trial.parameters['s'])
        
        # add observations once or multiple times
        study.add_observation(trial=trial,
                              iteration=i+1,
                              objective=pseudo_loss)

    study.finalize(trial=trial)
    print(pseudo_loss)


