# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 12:58:36 2021

@author: ATANIM
"""

from pyswmm import Simulation, Subcatchments, Nodes, Links
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
#%%
Sim = Simulation(r"C:\Users\ATANIM\Documents\Research\Rocky branch\Flood\Calibration.inp")
Sim.execute()
#%%
with Sim as sim:
    for step in sim:
        pass
    sim.report()

#%%

#%%

#%%
from jinja2 import Environment, FileSystemLoader
import logging

def render_input(tmp_folder,tmp_name,  data, out_inp):
    '''
    render an input file using a template.
    tmp_folder: folde where the template inp file is.
    tmp_name: name of the template file
    data: the data to be applied to the template
    out_inp: the inp file with values applied
    '''
    env = Environment(loader=FileSystemLoader(tmp_folder))
    template = env.get_template(tmp_name)
    output_from_parsed_template = template.render(data)
    # to save the results
    with open(out_inp, "w") as fh:
        fh.write(output_from_parsed_template)
        logging.info('inp created:%s' % out_inp)
#%%


def run(inp_path):
    sim = Simulation(inp_path)
    sim.execute()
#%%        
#from swmm.output import output as smo
#import pandas as pd
#import datetime


#%%
def swmm_dt(days):
    # convert swmm dates number to date
    # https://www.openswmm.org/Topic/4343/output-file-start-date-and-time-of-the-simulation
    t0 = pd.to_datetime('06/24/2020 00:00')
    t1 = t0 + datetime.timedelta(days=(days-1)) # I don't understand why I need to -1, hey it works. 
    return t1
#%%
import os
f= os.open(r'C:\Users\ATANIM\Documents\Research\Rocky branch\Flood\Calibration.inp', os.O_RDONLY)
open('f', 'r').read().find('')

#%%
lookup = '[SUBCATCHMENTS]'
with open(f) as myFile:
    for num, line in enumerate(myFile, 1):
        if lookup in line:
            print('found at line:', num)
lines = f.readlines() 
#%% step
#https://pypi.org/project/swmm-api/
#https://gitlab.com/markuspichler/swmm_api/-/tree/master
from swmm_api.input_file.section_labels import SUBCATCHMENTS,SUBAREAS, INFILTRATION, TRANSECTS
from swmm_api.input_file import read_inp_file, SwmmInput, section_labels as sections

inp = read_inp_file(r'C:\Users\ATANIM\Documents\Research\Rocky branch\Flood\Calibration.inp', convert_sections=[SUBCATCHMENTS,SUBAREAS, INFILTRATION, TRANSECTS])  # type: swmm_api.SwmmInput
#%%
from swmm_api import read_out_file


#%%
inp[sections.SUBCATCHMENTS]['ws-128'].Width = 286.6
inp[sections.SUBAREAS]['ws-128'].N_Imperv
inp[sections.SUBAREAS]['ws-128'].N_Perv
inp[sections.TRANSECTS]['c40'].roughness_channel
#%% step
def sub(x,a):
    inp[sections.SUBCATCHMENTS][x].Width = a 
    return a
#%% step
def cn(x,a):
    inp[sections.INFILTRATION][x].Psi = a 
    return a
#%% step
def Nimp(x,a):
    inp[sections.SUBAREAS][x].N_Imperv = a
    return a
#%% step
def Nperv(x,a):
    inp[sections.SUBAREAS][x].N_Perv = a
    return a
#%% step
catchment = ['ws-128', 'ws-148', 'ws-159', 'ws-71', 'ws-84', 'ws-161', 'ws-17']  
sw = []
sN = []
for i in catchment:
    rd = inp[sections.SUBCATCHMENTS][i].Width
    sw.append(rd)
    cd = inp[sections.INFILTRATION][i].Psi
    sN.append(cd)
    print(sw)
sw1 = []
sN1 = []
for i in sw:
    sa = 1.05*i
    sw1.append(sa)
for i in sN:
    ra = i+2
    sN1.append(ra)
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
def manch(x,a):
    inp[sections.TRANSECTS][x].roughness_channel = a
    return a
def manchr(x,a):
    inp[sections.TRANSECTS][x].roughness_right = a
    return a
def manchl(x,a):
    inp[sections.TRANSECTS][x].roughness_left = a
    return a
#%% step
#extract information on Nimp and Nper
nimp = []
nperv = []
for i in catchment:
    rd = inp[sections.SUBAREAS][i].N_Imperv
    nimp.append(rd)
    cd = inp[sections.SUBAREAS][i].N_Perv
    nperv.append(cd)
    print(nperv)
#%%
for i, j in zip(catchment, sw1):
    sub(i,j)

for i, j in zip(catchment, sN1):
    cn(i,j)
subcatchment= inp[sections.SUBCATCHMENTS].frame    
#%%
sw2 = [i*1.1 for i in sw]
sw3 = [i*1.15 for i in sw]
sw4 = [i*1.2 for i in sw]
sw5 = [i*1.25 for i in sw]
sw6 = [i*1.3 for i in sw]
sw7 = [i*1.5 for i in sw]
#%%
sN2 = [i+4 for i in sN]
sN3 = [i+6 for i in sN]
sN4 = [i+8 for i in sN]
sN5 = [i+10 for i in sN]
sN6 = [i+12 for i in sN]
sN7 = [i-5 for i in sN]
#%%
for i, j, k in zip(catchment, sw6, sN6):
    sub(i,j)
    cn(i,k)
    inp.write_file(r'C:\Users\ATANIM\Documents\Research\Rocky branch\Flood\new_inputfile5.inp')
#%%
import os
def inpf(sw,sn, fn):
    for i, j, k in zip(catchment, sw, sn):
        sub(i,j)
        cn(i,k)
        f_path = r'C:\Users\ATANIM\Documents\Research\Rocky branch\Flood'
        return inp.write_file(os.path.join(f_path, "%s.inp"% fn))
   
#inpf(sw,sN,"calr")
#store data in temporary input files
# re run the model
# determine the model accuracy
#%% step
import os
def inpf4(sw,sn,nimp, nperv, fn):
    for i, j, k, l, m in zip(catchment, sw, sn,nimp, nperv):
        sub(i,j)
        cn(i,k)
        Nimp(i,l)
        Nperv(i,m)
        f_path = r'C:\Users\ATANIM\Documents\Research\Rocky branch\Flood'
        return inp.write_file(os.path.join(f_path, "%s.inp"% fn))
#%%
def inpf7(sw,sn,nimp, nperv,n_chan, n_left, n_right, fn):
    for i, j, k, l, m in zip(catchment, sw, sn,nimp, nperv):
        sub(i,j)
        cn(i,k)
        Nimp(i,l)
        Nperv(i,m)
    for n, o, p, q in zip(Cond, n_chan, n_left, n_right):
        manch(n,o)
        manchr(n,p)
        manchl(n,q)
        
        f_path = r'C:\Users\ATANIM\Documents\Research\Rocky branch\Flood'
        return inp.write_file(os.path.join(f_path, "%s.inp"% fn))

#%%
with Simulation(r"C:\Users\ATANIM\Documents\Research\Rocky branch\Flood\Calibration.inp") as sim:
    # S1, S2 get the subcatchments by their names
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
#%%
df["S1 Discharge"].resample("15min").mean().plot(subplots=True, figsize=(15,4))
df["S2 Depth"].resample("15min").mean().plot(subplots=True, figsize=(15,4))   
#%%    
dis = df["S1 Discharge"].resample("5min").mean().to_frame()
dep = df["S2 Depth"].resample("15min").mean().to_frame()
df["S2 Depth"].resample("15min").mean().plot(subplots=True, figsize=(15,4))

vf_dep =  pd.read_csv(r'C:\Users\ATANIM\Documents\Research\Rocky branch\Validation\02169506.txt',  sep='\t', index_col='datetime')
d1 = dep["S2 Depth"].to_numpy()
d2 = ((vf_dep["depth"]*0.3048).to_numpy())#[:-2]
index = [0,1]
d3 = np.delete(d2, index)
r2_depth = ((np.corrcoef(d1, d3))[0,1])**2
#%% step
vf_dep =  pd.read_csv(r'C:\Users\ATANIM\Documents\Research\Rocky branch\Validation\02169506.txt',  sep='\t', index_col='datetime')
vf_dis = pd.read_csv(r'C:\Users\ATANIM\Documents\Research\Rocky branch\Validation\02169505.txt',  sep='\t', index_col='datetime')

#%% Determine the R2 value for water depth simulation
#mod = df["S2 Depth"], obs = vf_dep
def dep(mod, obs):
    dep = mod.resample("15min").mean().to_frame()
    d1 = dep["S2 Depth"].to_numpy()
    d2 = ((obs["depth"]*0.3048).to_numpy())#[:-2]
    n = d2.size - d1.size
    index = [i for i in range(0,n)]
    d3 = np.delete(d2, index)
    r2_depth = ((np.corrcoef(d1, d3))[0,1])**2
    return r2_depth
#%% Determine the R2 value for discharge simulation
def dis(mod, obs):
    dep = mod.resample("5min").mean().to_frame()
    d1 = dep["S1 Discharge"].to_numpy()
    d2 = ((obs["discharge"]*0.0283168).to_numpy())#[:-2]
    n = d2.size - d1.size
    index = [i for i in range(0,n)]
    d3 = np.delete(d2, index)
    #r2_dis = ((np.corrcoef(d1, d3))[0,1])**2
    nse = r2_score(d3, d1)
    return nse
#mod = df[S1 Discharge], obs = vf_dis
#%% trial
def runinp(fn):
    f_path = r'C:\Users\ATANIM\Documents\Research\Rocky branch\Flood'
    sim = Simulation(os.path.join(f_path, "%s.inp"% fn))
    sim.execute()
#%% trial
def model(sw,sn,fn):
    inpf(sw,sn,fn)
    return run(fn)
#%% create a SWMM input model by updating the model parameters
def inpmodel(sw,sn,fn):
    return inpf(sw,sn,fn)
     
#%% run a SWMM model to get the output depth and discharge at gauge location
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
inp1 = inp.write_file(r'C:\Users\ATANIM\Documents\Research\Rocky branch\Flood\new_inputfile.inp')
Sim1 = Simulation(r'C:\Users\ATANIM\Documents\Research\Rocky branch\Flood\new_inputfile.inp')
Sim1.execute()

#%%
#for n in range(0, 10):
#    globals()['list%s' % n] = list(range(n,n+3))
# run model the required functions are  input file, sub(x,a), cn(x,a), inpf, runf, dep, dep
inp = read_inp_file(r'C:\Users\ATANIM\Documents\Research\Rocky branch\Flood\Calibration.inp', convert_sections=[SUBCATCHMENTS, INFILTRATION])  # type: swmm_api.SwmmInput
inpf(sw, sN7, "cal2")

df = runf( "cal2")
dep(df["S2 Depth"],vf_dep)
dis(df["S1 Discharge"], vf_dis)

#%%


#%%
def modelswmm(a, b, fn):
    sw7 = [i*a for i in sw]
    sn2 = [i+b for i in sN]
    inpf(sw7, sn2, fn)
    df = runf(fn)
    r2 = dis(df["S1 Discharge"], vf_dis)
    return r2
#%% step
#Parameter senstivity regarding N-impervious and N-pervious
def modelswmm4(a, b, c, d, fn):
    sw7 = [i*a for i in sw]
    sn2 = [i+b for i in sN]
    nimp1 = [ i*c for i in nimp]
    nperv1 = [i*d for i in nperv]
    inpf4(sw7, sn2, nimp1, nperv1, fn)
    df = runf(fn)
    r2 = dis(df["S1 Discharge"], vf_dis)
    return r2
#%%
def modelswmm7(a, b, c, d,  e, f, g, fn):
    sw7 = [i*a for i in sw]
    sn2 = [i+b for i in sN]
    nimp1 = [ i*c for i in nimp]
    nperv1 = [i*d for i in nperv]
    n_chan1 = [i+e for i in n_chan]
    n_lef1 = [i+f for i in n_left]
    n_rig1  = [i+g for i in n_right]
    inpf7(sw7, sn2, nimp1, nperv1, n_chan1, n_lef1, n_rig1,  fn)
    df = runf(fn)
    r2 = dis(df["S1 Discharge"], vf_dis)
    return r2
#%%
def fr7(a, b, c, d, e, f, g):
    # print(params)  # <-- you'll see that params is a NumPy array
    # <-- for readability you may wish to assign names to the component variables
    r2 = modelswmm7(a, b, c, d,  e, f, g, "cal2")
    return r2
#%%
def ssfun(df):
    r2 = dis(df["S1 Discharge"], vf_dis)
    return 1-r2
#%%
import scipy.optimize as minimize
from numpy.random import rand

#%%
def fr(a, b):
    # print(params)  # <-- you'll see that params is a NumPy array
    # <-- for readability you may wish to assign names to the component variables
    r2 = modelswmm(a, b, "cal2")
    r3 = 1- r2
    return r2
#%% step
def fr4(a, b, c, d):
    # print(params)  # <-- you'll see that params is a NumPy array
    # <-- for readability you may wish to assign names to the component variables
    r2 = modelswmm4(a, b, c, d, "cal2")
    return r2
#%%
a = np.linspace(1.1, 2.5, 40)
b = np.linspace(1.00,10.00,40)

for i, j in zip (a,b):
    p = []
    p = modelswmm(i,j, "cal2")
    print(i,j,p)
#%% optimization function
from bayes_opt import BayesianOptimization

pbounds = {'a': (1, 2), 'b': (-5,10), 'c': (0.01,10), 'd': (0.01,1)}
           #'e': (-0.05,0.15), 'f': (-0.05,0.15), 'g': (-0.05,0.15) }

optimizer = BayesianOptimization(f = fr7, pbounds = pbounds, verbose =2,
     random_state =1
 )

optimizer.maximize(init_points =4,
                   n_iter =3
        )
#%%
from bayes_opt import BayesianOptimization

pbounds = {'a': (1, 2), 'b': (-5,10), 'c': (0.01,10), 'd': (0.01,1),
           'e': (-0.001,0.04), 'f': (-0.001,0.04), 'g': (-0.001,0.04) }

optimizer = BayesianOptimization(f = fr7, pbounds = pbounds, verbose =2,
     random_state =1
 )

optimizer.maximize(init_points =20,
                   n_iter =30
        )