# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 10:38:57 2022

@author: ATANIM
"""

from swmm_api.input_file.section_labels import SUBCATCHMENTS,SUBAREAS, INFILTRATION, TRANSECTS, OPTIONS
from swmm_api.input_file import read_inp_file, SwmmInput, section_labels as sections
import numpy as np
from pyswmm import Simulation, Subcatchments, Nodes, Links
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

inp = read_inp_file(r'C:\Users\ATANIM\Documents\Research\Rocky branch\Flood\Calibration_daily.inp', convert_sections=[SUBCATCHMENTS,SUBAREAS, INFILTRATION, TRANSECTS])  # type: swmm_api.SwmmInput

#%%
dict = {"fruits": 56}
def change_val(x,val):
    dict[x]  = val
    return val
change_val("fruits",30)
dict
#%%
def sub(x,a):
    inp[sections.SUBCATCHMENTS][x].Width = a 
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
catchment = ['ws-128', 'ws-148', 'ws-159', 'ws-71', 'ws-84', 'ws-161', 'ws-17']  
sw = []
slc = []
sN = []
Nimpl = []
Npervl = []
Simpl = []
Sprl = []
for i in catchment:
    rd = inp[sections.SUBCATCHMENTS][i].Width
    sw.append(rd)
    cd = inp[sections.INFILTRATION][i].Psi
    sN.append(cd)
    slci = inp[sections.SUBCATCHMENTS][i].Slope
    slc.append(slci)
    im = inp[sections.SUBAREAS][i].N_Imperv
    Nimpl.append(im)
    per = inp[sections.SUBAREAS][i].N_Perv
    Npervl.append(per)
    Sim = inp[sections.SUBAREAS][i].S_Imperv
    Simpl.append(Sim)
    Spr = inp[sections.SUBAREAS][i].S_Perv
    Sprl.append(Spr)
    print(Sprl)
    print(Simpl)
    print(sw)
    print(slc)
sw1 = []
sN1 = []
for i in sw:
    sa = 1.05*i
    sw1.append(sa)
for i in sN:
    ra = i+2
    sN1.append(ra)
#%%
vf_dis = pd.read_csv(r'C:\Users\ATANIM\Documents\Research\Rocky branch\Discharge data\discharge_19.csv')
vf_dis['Time'] =  pd.to_datetime(vf_dis['datetime'], format='%m/%d/%Y %H:%M')
vf_dis1 = vf_dis.set_index(['Time']).shift(0)
vf_dis1.pop('datetime')
vf_dis1 = vf_dis1.resample("D").mean().fillna(0)
#%%
vf_dep = pd.read_csv(r'C:\Users\ATANIM\Documents\Research\Rocky branch\Discharge data\depth_ver.csv')
vf_dep['Time'] = pd.to_datetime(vf_dep['datetime'], format='%m/%d/%Y %H:%M')
vf_dep1 = vf_dep.set_index(['Time']).shift(0)
vf_dep1.pop('datetime')
#%%
def dis(mod, obs):
    dep = mod.resample("D").mean().to_frame()
    d1 = dep["S1 Discharge"].to_numpy()
    d2 = ((obs["Discharge"]*0.02831).to_numpy())#[:-2]
    n = d2.size - d1.size
    index = [i for i in range(0,n)]
    d3 = np.delete(d2, index)
    #r2_dis = ((np.corrcoef(d1, d3))[0,1])**2
    nse = r2_score(d3, d1)
    return nse
#mod = df[S1 Discharge], obs = vf_dis
#%%
def dep(mod, obs):
    dep = mod.resample("D").mean().to_frame()
    d1 = dep["S2 Depth"].to_numpy()
    d2 = ((obs["Depth"]*0.3048).to_numpy())#[:-2]
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
def finpfi( Nimplr, Nperlv, slcl, swl, fn):
    for e,f in zip(catchment, Nimplr):
        Nimp(e,f)
    for g,h in zip(catchment,Nperlv):
        Nperv(g,h)
    for (m2,n2) in zip (catchment,slcl):
        sl(m2,n2)
    for (m3,n3) in zip (catchment,swl):
        sub(m3,n3)
        f_path = r'C:\Users\ATANIM\Documents\Research\Rocky branch\Flood'
        return inp.write_file(os.path.join(f_path, "%s.inp"% fn))
#%%
def fmodelswmmi(a1, b1, c1, d1, e1, f1, g1,
               a2, b2, c2, d2, e2, f2, g2,
               a5, b5, c5, d5, e5, f5, g5,
               a6, b6, c6, d6, e6, f6, g6, fn):
    n_imp = [Nimp('ws-128',(Nimpl[0]+a1)), Nimp('ws-148',(Nimpl[1]+b1)),Nimp('ws-159',(Nimpl[2]+c1)), Nimp('ws-71',(Nimpl[3]+d1)), Nimp('ws-84',(Nimpl[4]+e1)),
           Nimp('ws-161',(Nimpl[5]+f1)), Nimp('ws-17',(Nimpl[6]+g1))]
    n_per = [Nperv('ws-128',(Npervl[0]+a2)), Nperv('ws-148',(Npervl[1]+b2)),Nperv('ws-159',(Npervl[2]+c2)), Nperv('ws-71',(Npervl[3]+d2)), Nperv('ws-84',(Npervl[4]+e2)),
           Nperv('ws-161',(Npervl[5]+f2)), Nperv('ws-17',(Npervl[6]+g2))]
    slo = [sl('ws-128',(slc[0]+a5)), sl('ws-148',(slc[1]+b5)),sl('ws-159',(slc[2]+c5)), sl('ws-71',(slc[3]+d5)), sl('ws-84',(slc[4]+e5)),
           sl('ws-161',(slc[5]+f5)), sl('ws-17',(slc[6]+g5))]
    swo = [sub('ws-128',(sw[0]+a6)), sub('ws-148',(sw[1]+b6)),sub('ws-159',(sw[2]+c6)), sub('ws-71',(sw[3]+d6)), sub('ws-84',(sw[4]+e6)),
           sub('ws-161',(sw[5]+f6)), sub('ws-17',(sw[6]+g6))]    
    finpfi(n_imp, n_per, slo, swo, fn)
    df = runf(fn)
    r1 = dep(df["S2 Depth"],vf_dep1)
    r2 = dis(df["S1 Discharge"], vf_dis1)
    print(r1,r2)
    r = r1+r2
    return r
#%%
def fr28(a1, b1, c1, d1, e1, f1, g1,
               a2, b2, c2, d2, e2, f2, g2,
               a5, b5, c5, d5, e5, f5, g5,
               a6, b6, c6, d6, e6, f6, g6):
    # print(params)  # <-- you'll see that params is a NumPy array
    # <-- for readability you may wish to assign names to the component variables
    r2 = fmodelswmmi(a1, b1, c1, d1, e1, f1, g1,
               a2, b2, c2, d2, e2, f2, g2,
               a5, b5, c5, d5, e5, f5, g5,
               a6, b6, c6, d6, e6, f6, g6, "cal2")
    print(r2)
    return r2
#%%
import sherpa
parameters = [sherpa.Continuous(name='a1', range=[0.01,0.9]),
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
              sherpa.Continuous(name='a5', range=[-0.2*slc[0],+1.5*slc[0]]),
              sherpa.Continuous(name='b5', range=[-0.2*slc[1],+1.5*slc[1]]),
              sherpa.Continuous(name='c5', range=[-0.2*slc[2],+1.5*slc[2]]),
              sherpa.Continuous(name='d5', range=[-0.2*slc[3],+1.5*slc[3]]),
              sherpa.Continuous(name='e5', range=[-0.2*slc[4],+1.5*slc[4]]),
              sherpa.Continuous(name='f5', range=[-0.2*slc[5],+1.5*slc[5]]),
              sherpa.Continuous(name='g5', range=[-0.2*slc[6],+1.5*slc[6]]),
              sherpa.Continuous(name='a6', range=[-0.3*sw[0],+0.7*sw[0]]),
              sherpa.Continuous(name='b6', range=[-0.3*sw[1],+0.7*sw[1]]),
              sherpa.Continuous(name='c6', range=[-0.3*sw[2],+0.7*sw[2]]),
              sherpa.Continuous(name='d6', range=[-0.3*sw[3],+0.7*sw[3]]),
              sherpa.Continuous(name='e6', range=[-0.3*sw[4],+0.7*sw[4]]),
              sherpa.Continuous(name='f6', range=[-0.3*sw[5],+0.7*sw[5]]),
              sherpa.Continuous(name='g6', range=[-0.3*sw[6],+0.7*sw[6]])]
algorithm = sherpa.algorithms.GPyOpt(model_type='RF',
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
    num_iterations = 4
    for i in range(num_iterations):
        
        # access parameters via trial.parameters and id via trial.id
        pseudo_loss = fr28(trial.parameters['a1'],
                          trial.parameters['b1'], trial.parameters['c1'],
                          trial.parameters['d1'],trial.parameters['e1'],
                          trial.parameters['f1'],trial.parameters['g1'],
                           trial.parameters['a2'],
                          trial.parameters['b2'], trial.parameters['c2'],
                          trial.parameters['d2'],trial.parameters['e2'],
                          trial.parameters['f2'],trial.parameters['g2'],
                          trial.parameters['a5'],
                          trial.parameters['b5'], trial.parameters['c5'],
                          trial.parameters['d5'],trial.parameters['e5'],
                          trial.parameters['f5'],trial.parameters['g5'],
                           trial.parameters['a6'],
                          trial.parameters['b6'], trial.parameters['c6'],
                          trial.parameters['d6'],trial.parameters['e6'],
                          trial.parameters['f6'],trial.parameters['g6'])
        
        # add observations once or multiple times
        study.add_observation(trial=trial,
                              iteration=i+1,
                              objective=pseudo_loss)

    study.finalize(trial=trial)
    print(pseudo_loss)


