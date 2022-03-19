# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 13:10:20 2022

@author: ATANIM
"""

from swmm_api.input_file.section_labels import SUBCATCHMENTS,SUBAREAS, INFILTRATION, TRANSECTS
from swmm_api.input_file import read_inp_file, SwmmInput, section_labels as sections
import numpy as np
from pyswmm import Simulation, Subcatchments, Nodes, Links
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from datetime import datetime

inp = read_inp_file(r'C:\Users\ATANIM\Documents\Research\Rocky branch\Flood\Calibration_dcm.inp', convert_sections=[SUBCATCHMENTS,SUBAREAS, INFILTRATION, TRANSECTS])  # type: swmm_api.SwmmInput

#%%
def cn(x,a):
    inp[sections.INFILTRATION][x].Psi = a 
    return a
#%%
def Sperv(x,a):
    inp[sections.SUBAREAS][x].S_Perv = a
    return a
#%%
def Simperv(x,a):
    inp[sections.SUBAREAS][x].S_Imperv = a
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

sN = []
Simpl = []
Sprl = []
for i in catchment:
    cd = inp[sections.INFILTRATION][i].Psi
    sN.append(cd)
    Sim = inp[sections.SUBAREAS][i].S_Imperv
    Simpl.append(Sim)
    Spr = inp[sections.SUBAREAS][i].S_Perv
    Sprl.append(Spr)
    print(Sprl)
    print(Simpl)
    print(sN)
#%%
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
vf_dis = pd.read_csv(r'C:\Users\ATANIM\Documents\Research\Rocky branch\Validation\02169505CALdis.txt', sep='\t')
vf_dis['Time'] =  pd.to_datetime(vf_dis['datetime'], format='%m/%d/%Y %H:%M')
vf_dis1 = vf_dis.set_index(['Time']).shift(0)
vf_dis1.pop('datetime')
vf_dis1 = vf_dis1.resample("0.25H").mean().fillna(0)
#%%
vf_dep = pd.read_csv(r'C:\Users\ATANIM\Documents\Research\Rocky branch\Validation\02169506depth.txt', sep='\t')
vf_dep['datetime'] = pd.to_datetime(vf_dep['datetime'], format='%m/%d/%Y %H:%M')
vf_dep1 = vf_dep.set_index(['datetime']).shift(0)
#%%
def dis(mod, obs):
    dep = mod.resample("0.25H").mean().to_frame()
    d1 = dep["S1 Discharge"].to_numpy()
    d2 = ((obs["discharge"]*0.02831).to_numpy())#[:-2]
    n = d2.size - d1.size
    index = [i for i in range(0,n)]
    d3 = np.delete(d2, index)
    #r2_dis = ((np.corrcoef(d1, d3))[0,1])**2
    nse = r2_score(d3, d1)
    return nse
#mod = df[S1 Discharge], obs = vf_dis
#%%
def dep(mod, obs):
    dep = mod.resample("0.25H").mean().to_frame()
    d1 = dep["S2 Depth"].to_numpy()
    d2 = ((obs["depth"]*0.3048).to_numpy())#[:-2]
    n = d2.size - d1.size
    index = [i for i in range(0,n)]
    d3 = np.delete(d2, index)
    r2_depth = ((np.corrcoef(d1, d3))[0,1])**2
    return r2_depth
#%%
import os
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
def inpfi(cnu, Simplr, Spervl, n_chan, n_right, n_left, fn):
    for i,j in zip(catchment, cnu):
        cn(i,j)
    for m,n in zip(catchment,Simplr):
        Simperv(m,n)
    for m1,n1 in zip(catchment,Spervl):
        Sperv(m1,n1)
    for k, l in zip(Cond, n_chan):
        manch(k,l)
    for a, b in zip(Cond, n_right):
        manchr(a, b)
    for c, d in zip(Cond, n_left):
        manchr(c, d)
        f_path = r'C:\Users\ATANIM\Documents\Research\Rocky branch\Flood'
        return inp.write_file(os.path.join(f_path, "%s.inp"% fn))
#%%
def dmodelswmmi(a,b,c,d,e,f,g,
               a3, b3, c3, d3, e3, f3, g3,
               a4, b4, c4, d4, e4, f4, g4,
               i,j,k,l, h, m, n,o, p, q, r, s, fn):
    CNU = [cn('ws-128',(sN[0]+a)), cn('ws-148',(sN[1]+b)),cn('ws-159',(sN[2]+c)), cn('ws-71',(sN[3]+d)), cn('ws-84',(sN[4]+e)),
           cn('ws-161',(sN[5]+f)), cn('ws-17',(sN[6]+g))]
    S_imper =[Simperv('ws-128',(Simpl[0]+a3)), Simperv('ws-148',(Simpl[1]+b3)),Simperv('ws-159',(Simpl[2]+c3)), Simperv('ws-71',(Simpl[3]+d3)), Simperv('ws-84',(Simpl[4]+e3)),
           Simperv('ws-161',(Simpl[5]+f3)), Simperv('ws-17',(Simpl[6]+g3))]
    S_per = [Sperv('ws-128',(Sprl[0]+a4)), Sperv('ws-148',(Sprl[1]+b4)),Sperv('ws-159',(Sprl[2]+c4)), Sperv('ws-71',(Sprl[3]+d4)), Sperv('ws-84',(Sprl[4]+e4)),
           Sperv('ws-161',(Sprl[5]+f4)), Sperv('ws-17',(Sprl[6]+g4))]
    n_ch = [manch('c67',(n_chan[0]+i)),(manch('c68',n_chan[1])+j), (manch('c70',n_chan[2])+k), (manch('c71',n_chan[3])+l)]
    n_r = [manchr('c67',(n_right[0]+h)), manchr('c68',(n_right[1]+m)), manchr('c70',(n_right[2]+n)), manchr('c71',(n_right[3]+o))]
    n_l = [manchl('c67',(n_left[0]+p)), manchl('c68',(n_left[1]+q)), manchl('c70',(n_left[2]+r)), manchl('c71',(n_left[3]+s))]
    inpfi(CNU, S_imper,S_per, n_ch,n_r,n_l, fn)
    df = runf(fn)
    r1 = dep(df["S2 Depth"],vf_dep1)
    r2 = dis(df["S1 Discharge"], vf_dis1)
    print(r1,r2)
    r = r1+r2
    return r
#%%
def fr33(a,b,c,d,e,f,g,
               a3, b3, c3, d3, e3, f3, g3,
               a4, b4, c4, d4, e4, f4, g4,
               i,j,k,l, h, m, n,o, p, q, r, s):
    # print(params)  # <-- you'll see that params is a NumPy array
    # <-- for readability you may wish to assign names to the component variables
    r2 = dmodelswmmi(a,b,c,d,e,f,g,
               a3, b3, c3, d3, e3, f3, g3,
               a4, b4, c4, d4, e4, f4, g4,
               i,j,k,l, h, m, n,o, p, q, r, s, "cal2")
    print(r2)
    return r2
#%%
import sherpa
parameters = [sherpa.Continuous(name='a', range=[-5,10]),
              sherpa.Continuous(name='b', range=[-6,10]),
              sherpa.Continuous(name='c', range=[-6,10]),
              sherpa.Continuous(name='d', range=[-6,10]),
              sherpa.Continuous(name='e', range=[-6,10]),
              sherpa.Continuous(name='f', range=[-6,10]),
              sherpa.Continuous(name='g', range=[-6,10]),       
              sherpa.Continuous(name='a3', range=[-0.5,5.5]),
              sherpa.Continuous(name='b3', range=[-0.5,5.5]),
              sherpa.Continuous(name='c3', range=[-0.5,5.5]),
              sherpa.Continuous(name='d3', range=[-0.5,5.5]),
              sherpa.Continuous(name='e3', range=[-0.5,5.5]),
              sherpa.Continuous(name='f3', range=[-0.5,5.5]),
              sherpa.Continuous(name='g3', range=[-0.5,5.5]),
              sherpa.Continuous(name='a4', range=[-2.5,3.5]),
              sherpa.Continuous(name='b4', range=[-2.5,3.5]),
              sherpa.Continuous(name='c4', range=[-2.5,3.5]),
              sherpa.Continuous(name='d4', range=[-2.5,3.5]),
              sherpa.Continuous(name='e4', range=[-2.5,3.5]),
              sherpa.Continuous(name='f4', range=[-2.5,3.5]),
              sherpa.Continuous(name='g4', range=[-2.5,3.5]),
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
                                     acquisition_type='EI', 
                                     max_concurrent=1)
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
                          trial.parameters['g'], 
                          trial.parameters['a3'],
                          trial.parameters['b3'], trial.parameters['c3'],
                          trial.parameters['d3'],trial.parameters['e3'],
                          trial.parameters['f3'],trial.parameters['g3'],
                          trial.parameters['a4'],
                          trial.parameters['b4'], trial.parameters['c4'],
                          trial.parameters['d4'],trial.parameters['e4'],
                          trial.parameters['f4'],trial.parameters['g4'],
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
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))