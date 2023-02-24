# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 13:30:45 2022

@author: ATANIM
"""
from swmm_api.input_file.section_labels import SUBCATCHMENTS,SUBAREAS, INFILTRATION, TRANSECTS, OPTIONS, DWF
from swmm_api.input_file import read_inp_file, SwmmInput, section_labels as sections
import numpy as np
from pyswmm import Simulation, Subcatchments, Nodes, Links
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

inp = read_inp_file(r'C:\Users\ATANIM\Documents\Research\Rocky branch\Flood\Calibration_rtc22_3.inp', convert_sections=[SUBCATCHMENTS,SUBAREAS, INFILTRATION, TRANSECTS, DWF])  # type: swmm_api.SwmmInput

#%%here input file is a dictionary we have to create a function to update the dictionary
def cn(x,a):
    inp[sections.INFILTRATION][x].Psi = a # HERE the value 'a' will be assigned to the corresponding parameters in  the dictionary
    return a# it may seem  be 
#%%
def Sperv(x,a):
    inp[sections.SUBAREAS][x].S_Perv = a
    return a
#%%
def Simperv(x,a):
    inp[sections.SUBAREAS][x].S_Imperv = a
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

sN = []
Nimpl = []
Npervl = []
Simpl = []
Sprl = []
for i in catchment:
    cd = inp[sections.INFILTRATION][i].Psi
    sN.append(cd)
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
#%% see
ri = inp[sections.OPTIONS]

for item in ri.split("\n"):
    if "START_DATE           " in item:
        ri1 = ""
        ri1= ri1+ item.strip()
ri1.replace(ri1[21:31], '09/02/2018')
        
ri.replace("START_DATE           09/01/2018","START_DATE           09/02/2018")
#%%
# a function to change the start and end date of the swmm model
def rep(ri, txt1,txt1x, txt2,txt2x, txt3, txt4):
    for item in ri.split("\n"):
        if "START_DATE           " in item:
            ri1 = ""
            ri1= ri1+ item.strip()
            ri2 = ri1.replace(ri1[21:31], txt1)
            rin= ri.replace(ri1,ri2)
        elif "REPORT_START_DATE" in item:
            rit1 = ""
            rit1= rit1+ item.strip()
            rit2 = rit1.replace(rit1[21:31], txt1x)
            rin= rin.replace(rit1,rit2)
        elif "START_TIME           " in item:
            rsp1 = ""
            rsp1= rsp1+ item.strip()
            rsp2 = rsp1.replace(rsp1[21:29], txt2)
            rin= rin.replace(rsp1,rsp2)
        elif "REPORT_START_TIME    " in item:
            rp1 = ""
            rp1= rp1+ item.strip()
            rp2 = rp1.replace(rp1[21:29], txt2x)
            rin= rin.replace(rp1,rp2)
        elif "END_DATE" in item:
            rf1 = ""
            rf1= rf1+ item.strip()
            rf2 = rf1.replace(rf1[21:31], txt3)
            rin= rin.replace(rf1,rf2)
        elif "END_TIME" in item:
            rft1 = ""
            rft1= rft1+ item.strip()
            rft2 = rft1.replace(rft1[21:29], txt4)
            rin= rin.replace(rft1,rft2)
    return rin
#%%
#import re
#from datetime import datetime, timedelta
#itx = inp[sections.OPTIONS]
#rl1 = ""
#rd1 = ""
#rdi1 = ""
# =============================================================================
# #%%
# for item in itx.split("\n"):
#     if re.findall("START_TIME           ", item):# re.finall returns exact case search. There are many instances of the search case
#             rl1= ""+ item.strip()
#     elif "START_DATE           " in item:
#             rd1= ""+ item.strip()
#             rdi1 = rd1[21:31] + " " + rl1[21:31]
#             rdi1 = datetime.strptime(rdi1, '%m/%d/%Y %H:%M:%S')
# #%%
# starttime_st = rdi1.strftime("%m/%d/%Y %H:%M:%S")
# startdate= starttime_st[0:10]
# startime = starttime_st[11:19]
# stoptime_f = rdi1+ timedelta(hours=24, minutes=00, seconds=00)# add the time steps here
# stoptime_st= stoptime_f.strftime("%m/%d/%Y %H:%M:%S")
# reporttime_f = stoptime_f - timedelta(hours=4, minutes=00, seconds=00)
# reporttime_st = reporttime_f.strftime("%m/%d/%Y %H:%M:%S")
# re_startdate = reporttime_st[0:10]
# re_startime = reporttime_st[11:19]
# stopdate= stoptime_st[0:10]
# stoptime = stoptime_st[11:19]
# inp[sections.OPTIONS] = rep(inp[sections.OPTIONS], startdate, startdate, startime, startime, stopdate, stoptime)
# =============================================================================
#%%
vf_dis = pd.read_csv(r'E:\Rocky Branch\Observed data\24_Aug_Pickens_St_dis.csv')
vf_dis['Time'] =  pd.to_datetime(vf_dis['datetime'], format='%m/%d/%Y %H:%M')
vf_dis1 = vf_dis.set_index(['Time']).shift(0)
vf_dis1.pop('datetime')
inp[sections.DWF][('159', 'FLOW')].Base = vf_dis1.iloc[1, 0]*0.02831
#vf_dis1 = vf_dis1.resample("D").mean().fillna(0)
#%%
vf_dep = pd.read_csv(r'E:\Rocky Branch\Observed data\24_Aug_whaley_St.csv')
vf_dep['Time'] = pd.to_datetime(vf_dep['datetime'], format='%m/%d/%Y %H:%M')
vf_dep1 = vf_dep.set_index(['Time']).shift(0)
vf_dep1.pop('datetime')
#%%
vf_dep_sen = pd.read_csv(r'E:\Rocky Branch\Observed data\olympia_park_08_24.csv')#sensor data
vf_dep_sen['Time'] = pd.to_datetime(vf_dep_sen['datetime'], format='%m/%d/%Y %H:%M')
vf_dep_s1 = vf_dep_sen.set_index(['Time']).shift(0)
vf_dep_s1 = vf_dep_s1.resample("0.25H").mean()
#vf_dep_s1.pop('datetime')

#%%
#list1=[]
#list2=[]
#list3=[]
#list4=[]
#for i in range(len(list1)):
#    inp[sections.OPTIONS] = rep(inp[sections.OPTIONS], list1[i], list2[i], list3[i], list4[i])

#update the SWMM options
#inp[sections.OPTIONS] = rep(inp[sections.OPTIONS], '08/24/2022', '16:3','08/25/2022','14:0')
# slice a dataframe based on start and ending point
vf_dis1= vf_dis1.loc["8/25/2022 19:30":"8/25/2022 22:00"]
vf_dep1= vf_dep1.loc["8/25/2022 19:30":"8/25/2022 22:00"]
vf_dep_s = vf_dep_s1.loc["8/25/2022 19:30":"8/25/2022 22:00"]
#%%
# now create an input file for the model set up for the retrieved datasets
#create a function that returns the input file and observed dis depth







#%%
def dis(mod, obs):
    dep = mod.resample("0.25H").mean().to_frame()
    #dep= dep.loc["8/25/2022 06:30":"8/25/2022 15:30"]### define the ranage
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
    #dep= dep.loc["8/25/2022 06:30":"8/25/2022 15:30"]### define the ranage
    d1 = dep["S2 Depth"].to_numpy()
    d2 = ((obs["depth"]*0.3048).to_numpy())#[:-2]
    n = d2.size - d1.size
    index = [i for i in range(0,n)]
    d3 = np.delete(d2, index)
    r2_depth = ((np.corrcoef(d1, d3))[0,1])**2
    return r2_depth

#%%
def dep1(mod, obs):
    dep = mod.resample("0.25H").mean().to_frame()
    #dep= dep.loc["8/25/2022 06:30":"8/25/2022 15:30"]### define the ranage
    d1 = dep["U1 Depth"].to_numpy()
    d2 = ((obs["depth"]).to_numpy())#[:-2]
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
        S3 = Nodes(sim)["15"] 
    # we need to create a time series table, idx is the time, s1_values, s2_values are the two columns for the runoff of s1 and s1
        idx = []
        s1_values = []
        s2_values = []
        s3_values = []
    # as we are looping through each time step, we add the simulated value into the 3 coloumns variables above
        for step in sim:
            idx.append(sim.current_time)
            s1_values.append(S1.flow)
            s2_values.append(S2.depth)
            s3_values.append(S3.depth)
    # using this line below, we turn the 3 columns into a table, called DataFrame using the pandas library. So that we can plot it.
        df = pd.DataFrame({'S1 Discharge': s1_values, 'S2 Depth': s2_values, 'U1 Depth': s3_values}, index=idx)
        df1= df.loc["8/25/2022 19:30":"8/25/2022 22:00"]#report start and end time
        return df1
#%%
def inpfi(cnu, Nimplr, Nperlv, Simplr, Spervl, n_chan, n_right, n_left, fn):
    for i,j in zip(catchment, cnu):
        cn(i,j)
    for e,f in zip(catchment, Nimplr):
        Nimp(e,f)
    for g,h in zip(catchment,Nperlv):
        Nperv(g,h)
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
               a1, b1, c1, d1, e1, f1, g1,
               a2, b2, c2, d2, e2, f2, g2,
               a3, b3, c3, d3, e3, f3, g3,
               a4, b4, c4, d4, e4, f4, g4,
               i,j,k,l, h, m, n,o, p, q, r, s, fn):
    CNU = [cn('ws-128',(sN[0]+a)), cn('ws-148',(sN[1]+b)),cn('ws-159',(sN[2]+c)), cn('ws-71',(sN[3]+d)), cn('ws-84',(sN[4]+e)),
           cn('ws-161',(sN[5]+f)), cn('ws-17',(sN[6]+g))]
    n_imp = [Nimp('ws-128',(Nimpl[0]+a1)), Nimp('ws-148',(Nimpl[1]+b1)),Nimp('ws-159',(Nimpl[2]+c1)), Nimp('ws-71',(Nimpl[3]+d1)), Nimp('ws-84',(Nimpl[4]+e1)),
           Nimp('ws-161',(Nimpl[5]+f1)), Nimp('ws-17',(Nimpl[6]+g1))]
    n_per = [Nperv('ws-128',(Npervl[0]+a2)), Nperv('ws-148',(Npervl[1]+b2)),Nperv('ws-159',(Npervl[2]+c2)), Nperv('ws-71',(Npervl[3]+d2)), Nperv('ws-84',(Npervl[4]+e2)),
           Nperv('ws-161',(Npervl[5]+f2)), Nperv('ws-17',(Npervl[6]+g2))]
    S_imper =[Simperv('ws-128',(Simpl[0]+a3)), Simperv('ws-148',(Simpl[1]+b3)),Simperv('ws-159',(Simpl[2]+c3)), Simperv('ws-71',(Simpl[3]+d3)), Simperv('ws-84',(Simpl[4]+e3)),
           Simperv('ws-161',(Simpl[5]+f3)), Simperv('ws-17',(Simpl[6]+g3))]
    S_per = [Sperv('ws-128',(Sprl[0]+a4)), Sperv('ws-148',(Sprl[1]+b4)),Sperv('ws-159',(Sprl[2]+c4)), Sperv('ws-71',(Sprl[3]+d4)), Sperv('ws-84',(Sprl[4]+e4)),
           Sperv('ws-161',(Sprl[5]+f4)), Sperv('ws-17',(Sprl[6]+g4))]
    n_ch = [manch('c67',(n_chan[0]+i)),(manch('c68',n_chan[1])+j), (manch('c70',n_chan[2])+k), (manch('c71',n_chan[3])+l)]
    n_r = [manchr('c67',(n_right[0]+h)), manchr('c68',(n_right[1]+m)), manchr('c70',(n_right[2]+n)), manchr('c71',(n_right[3]+o))]
    n_l = [manchl('c67',(n_left[0]+p)), manchl('c68',(n_left[1]+q)), manchl('c70',(n_left[2]+r)), manchl('c71',(n_left[3]+s))]
    inpfi(CNU, n_imp, n_per, S_imper,S_per, n_ch,n_r,n_l, fn)
    df = runf(fn)
    r1 = dep(df["S2 Depth"],vf_dep1)
    r2 = dis(df["S1 Discharge"], vf_dis1)
    r3 = dep1(df["U1 Depth"], vf_dep_s1)
    print(r1,r2, r3)
    r = 1*r1+1*r2+ 0.0*r3
    return r
#%%
def fr33(a,b,c,d,e,f,g,
         a1, b1, c1, d1, e1, f1, g1,
               a2, b2, c2, d2, e2, f2, g2,
               a3, b3, c3, d3, e3, f3, g3,
               a4, b4, c4, d4, e4, f4, g4,
               i,j,k,l, h, m, n,o, p, q, r, s):
    # print(params)  # <-- you'll see that params is a NumPy array
    # <-- for readability you may wish to assign names to the component variables
    r2 = dmodelswmmi(a,b,c,d,e,f,g,
            a1, b1, c1, d1, e1, f1, g1,
               a2, b2, c2, d2, e2, f2, g2,
               a3, b3, c3, d3, e3, f3, g3,
               a4, b4, c4, d4, e4, f4, g4,
               i,j,k,l, h, m, n,o, p, q, r, s, "cal2")
    print(r2)
    return r2

#%%
import sherpa
from datetime import datetime
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
              sherpa.Continuous(name='b2', range=[-0.01,0.6]),
              sherpa.Continuous(name='c2', range=[-0.01,0.6]),
              sherpa.Continuous(name='d2', range=[-0.45,0.4]),
              sherpa.Continuous(name='e2', range=[-0.35,0.6]),
              sherpa.Continuous(name='f2', range=[-0.35,0.6]),
              sherpa.Continuous(name='g2', range=[-0.35,0.6]),
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
                                     acquisition_type='MPI',
                                     num_initial_data_points='infer',
                                     max_concurrent=1,
                                     verbosity = 10,
                                     max_num_trials =100)
#algorithm = sherpa.algorithms.PopulationBasedTraining(population_size=10,
#                                                      num_generations=5,
#                                                      perturbation_factors=(0.8, 1.2))

study = sherpa.Study(parameters=parameters,
                     algorithm=algorithm,
                     lower_is_better=False,
                     disable_dashboard=True)
#result = open(r'C:\Users\ATANIM\Documents\Research\Rocky branch\Calibration\optimization.txt', 'w')
#result.close()

tL = []
Z = []
a = []
b = []
c = []
d = []
e = []
f = []
g = []
a1 = []
b1 = []
c1= []
d1 = []
e1 = []
f1 = []
g1 = []    
a2 = []
b2 = []
c2 = []
d2 = []
e2 = []
f2 = []
g2 = []
a3 = []
b3 = []
c3 = []
d3 = []
e3 = []
f3 = []
g3 = []
a4 = []
b4 = []
c4 = []
d4 = []
e4 = []
f4 = []
g4 = []
i = []
j = []
k = []
l = []
h = []
m = []
n = []
o = []
p = []
q = []
r = []
s = []
for trial in study:
    print("Trial {}:\t{}".format(trial.id, trial.parameters))

    num_iterations = 1
    for num in range(num_iterations):
        
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
                              iteration=num+1,
                              objective=pseudo_loss)

    study.finalize(trial=trial)
    a.append(trial.parameters['a'])
    b.append(trial.parameters['b'])
    c.append(trial.parameters['c'])
    d.append(trial.parameters['d'])
    e.append(trial.parameters['e'])
    f.append(trial.parameters['f'])
    g.append(trial.parameters['g'])
    a1.append(trial.parameters['a1'])
    b1.append(trial.parameters['b1'])
    c1.append(trial.parameters['c1'])
    d1.append(trial.parameters['d1'])
    e1.append(trial.parameters['e1'])
    f1.append(trial.parameters['f1'])
    g1.append(trial.parameters['g1'])   
    a2.append(trial.parameters['a2'])
    b2.append(trial.parameters['b2'])
    c2.append(trial.parameters['c2'])
    d2.append(trial.parameters['d2'])
    e2.append(trial.parameters['e2'])
    f2.append(trial.parameters['f2'])
    g2.append(trial.parameters['g2'])
    a3.append(trial.parameters['a3'])
    b3.append(trial.parameters['b3'])
    c3.append(trial.parameters['c3'])
    d3.append(trial.parameters['d3'])
    e3.append(trial.parameters['e3'])
    f3.append(trial.parameters['f3'])
    g3.append(trial.parameters['g3'])
    a4.append(trial.parameters['a4'])
    b4.append(trial.parameters['b4'])
    c4.append(trial.parameters['c4'])
    d4.append(trial.parameters['d4'])
    e4.append(trial.parameters['e4'])
    f4.append(trial.parameters['f4'])
    g4.append(trial.parameters['g4'])
    i.append(trial.parameters['i'])
    j.append(trial.parameters['j'])
    k.append(trial.parameters['k'])
    l.append(trial.parameters['l'])
    h.append(trial.parameters['h'])
    m.append(trial.parameters['m'])
    n.append(trial.parameters['n'])
    o.append(trial.parameters['o'])
    p.append(trial.parameters['p'])
    q.append(trial.parameters['q'])
    r.append(trial.parameters['r'])
    s.append(trial.parameters['s'])
    print(pseudo_loss)
    Z.append(pseudo_loss)
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    tL.append(now)
    print(now)
#%%
max_id, max_no = max(enumerate(Z), key=lambda x: x[1])
fr33(a[max_id],b[max_id],c[max_id],d[max_id],e[max_id],f[max_id],g[max_id], 
     a1[max_id],b1[max_id],c1[max_id],d1[max_id],e1[max_id],f1[max_id],g1[max_id],
     a2[max_id],b2[max_id],c2[max_id],d2[max_id],e2[max_id],f2[max_id],g2[max_id],
     a3[max_id], b3[max_id], c3[max_id], d3[max_id], e3[max_id], f3[max_id], g3[max_id],
               a4[max_id], b4[max_id], c4[max_id], d4[max_id], e4[max_id], f4[max_id], g4[max_id],
               i[max_id],j[max_id],k[max_id],l[max_id], h[max_id], m[max_id], n[max_id],o[max_id], 
               p[max_id], q[max_id], r[max_id], s[max_id])
p_best = [a[max_id],b[max_id],c[max_id],d[max_id],e[max_id],f[max_id],g[max_id],
     a1[max_id],b1[max_id],c1[max_id],d1[max_id],e1[max_id],f1[max_id],g1[max_id],
     a2[max_id],b2[max_id],c2[max_id],d2[max_id],e2[max_id],f2[max_id],g2[max_id],          
          a3[max_id], b3[max_id], c3[max_id], d3[max_id], e3[max_id], f3[max_id], g3[max_id],
               a4[max_id], b4[max_id], c4[max_id], d4[max_id], e4[max_id], f4[max_id], g4[max_id],
               i[max_id],j[max_id],k[max_id],l[max_id], h[max_id], m[max_id], n[max_id],o[max_id], 
               p[max_id], q[max_id], r[max_id], s[max_id]]
plist9 = pd.DataFrame(p_best)
plist9.to_csv(r'E:\Rocky Branch\da\plist9.csv', header = True, index=True)
#%%
#inp[sections.OPTIONS] = rep(inp[sections.OPTIONS], '08/25/2022', '14:0','08/25/2022','22:0')
# slice a dataframe based on start and ending point
#%% re-initiate the depth discharge file
# =============================================================================
# vf_dis1= vf_dis1.loc["8/25/2022 14:00":"8/25/2022 22:00"]
# vf_dep1= vf_dep1.loc["8/25/2022 14:00":"8/25/2022 22:00"]
# fr33(a[max_id],b[max_id],c[max_id],d[max_id],e[max_id],f[max_id],g[max_id], 
#      a1[max_id],b1[max_id],c1[max_id],d1[max_id],e1[max_id],f1[max_id],g1[max_id],
#      a2[max_id],b2[max_id],c2[max_id],d2[max_id],e2[max_id],f2[max_id],g2[max_id],
#      a3[max_id], b3[max_id], c3[max_id], d3[max_id], e3[max_id], f3[max_id], g3[max_id],
#                a4[max_id], b4[max_id], c4[max_id], d4[max_id], e4[max_id], f4[max_id], g4[max_id],
#                i[max_id],j[max_id],k[max_id],l[max_id], h[max_id], m[max_id], n[max_id],o[max_id], 
#                p[max_id], q[max_id], r[max_id], s[max_id])
# =============================================================================
#%%
# =============================================================================
# f_path = r'C:\Users\ATANIM\Documents\Research\Rocky branch\Flood'
# sim_path = os.path.join(f_path, "cal2.inp")
# with Simulation(sim_path) as sim:
#     S1 = Links(sim)["usgs_dis"]
#     S2 = Nodes(sim)["43_depth"]   
# # we need to create a time series table, idx is the time, s1_values, s2_values are the two columns for the runoff of s1 and s1
#     idx = []
#     s1_values = []
#     s2_values = []   
# # as we are looping through each time step, we add the simulated value into the 3 coloumns variables above
#     sim.step_advance(150)
#     for step in sim:
#           idx.append(sim.current_time)
#           s1_values.append(S1.flow)
#           s2_values.append(S2.depth)
#           
#           dt_hs_file = 'tmp_hsf.hsf'
#           dt_hs_path = os.path.join(f_path, dt_hs_file)
#           #sim.save_hotstart(dt_hs_path)
# # using this line below, we turn the 3 columns into a table, called DataFrame using the pandas library. So that we can plot it.
#     df = pd.DataFrame({'S1 Discharge': s1_values, 'S2 Depth': s2_values}, index=idx)
# =============================================================================
#%%
#Test2
#from swmmio.run_models.run import run_simple, run_hot_start_sequence
#run_simple(r'C:\Users\ATANIM\Documents\Research\Rocky branch\Flood\Calibration_rtc2022.inp')
#run_hot_start_sequence(r'C:\Users\ATANIM\Documents\Research\Rocky branch\Flood\Calibration_rtc2022.inp')

#%%
def dfmswmmi(a,b,c,d,e,f,g,
               a1, b1, c1, d1, e1, f1, g1,
               a2, b2, c2, d2, e2, f2, g2,
               a3, b3, c3, d3, e3, f3, g3,
               a4, b4, c4, d4, e4, f4, g4,
               i,j,k,l, h, m, n,o, p, q, r, s, fn):
    CNU = [cn('ws-128',(sN[0]+a)), cn('ws-148',(sN[1]+b)),cn('ws-159',(sN[2]+c)), cn('ws-71',(sN[3]+d)), cn('ws-84',(sN[4]+e)),
           cn('ws-161',(sN[5]+f)), cn('ws-17',(sN[6]+g))]
    n_imp = [Nimp('ws-128',(Nimpl[0]+a1)), Nimp('ws-148',(Nimpl[1]+b1)),Nimp('ws-159',(Nimpl[2]+c1)), Nimp('ws-71',(Nimpl[3]+d1)), Nimp('ws-84',(Nimpl[4]+e1)),
           Nimp('ws-161',(Nimpl[5]+f1)), Nimp('ws-17',(Nimpl[6]+g1))]
    n_per = [Nperv('ws-128',(Npervl[0]+a2)), Nperv('ws-148',(Npervl[1]+b2)),Nperv('ws-159',(Npervl[2]+c2)), Nperv('ws-71',(Npervl[3]+d2)), Nperv('ws-84',(Npervl[4]+e2)),
           Nperv('ws-161',(Npervl[5]+f2)), Nperv('ws-17',(Npervl[6]+g2))]
    S_imper =[Simperv('ws-128',(Simpl[0]+a3)), Simperv('ws-148',(Simpl[1]+b3)),Simperv('ws-159',(Simpl[2]+c3)), Simperv('ws-71',(Simpl[3]+d3)), Simperv('ws-84',(Simpl[4]+e3)),
           Simperv('ws-161',(Simpl[5]+f3)), Simperv('ws-17',(Simpl[6]+g3))]
    S_per = [Sperv('ws-128',(Sprl[0]+a4)), Sperv('ws-148',(Sprl[1]+b4)),Sperv('ws-159',(Sprl[2]+c4)), Sperv('ws-71',(Sprl[3]+d4)), Sperv('ws-84',(Sprl[4]+e4)),
           Sperv('ws-161',(Sprl[5]+f4)), Sperv('ws-17',(Sprl[6]+g4))]
    n_ch = [manch('c67',(n_chan[0]+i)),(manch('c68',n_chan[1])+j), (manch('c70',n_chan[2])+k), (manch('c71',n_chan[3])+l)]
    n_r = [manchr('c67',(n_right[0]+h)), manchr('c68',(n_right[1]+m)), manchr('c70',(n_right[2]+n)), manchr('c71',(n_right[3]+o))]
    n_l = [manchl('c67',(n_left[0]+p)), manchl('c68',(n_left[1]+q)), manchl('c70',(n_left[2]+r)), manchl('c71',(n_left[3]+s))]
    inpfi(CNU, n_imp, n_per, S_imper,S_per, n_ch,n_r,n_l, fn)
    df = runf(fn)
    dfr = df.resample("0.25H").mean()
    return dfr
#%%
dfitlr9 = dfmswmmi(a[max_id],b[max_id],c[max_id],d[max_id],e[max_id],f[max_id],g[max_id], 
     a1[max_id],b1[max_id],c1[max_id],d1[max_id],e1[max_id],f1[max_id],g1[max_id],
     a2[max_id],b2[max_id],c2[max_id],d2[max_id],e2[max_id],f2[max_id],g2[max_id],
     a3[max_id], b3[max_id], c3[max_id], d3[max_id], e3[max_id], f3[max_id], g3[max_id],
               a4[max_id], b4[max_id], c4[max_id], d4[max_id], e4[max_id], f4[max_id], g4[max_id],
               i[max_id],j[max_id],k[max_id],l[max_id], h[max_id], m[max_id], n[max_id],o[max_id], 
               p[max_id], q[max_id], r[max_id], s[max_id], "cal2")
dfitlr9.to_csv(r'E:\Rocky Branch\da\dfitlr9.csv', header = True, index=True)
fig, axes = plt.subplots(1, 1, figsize=(10, 5), sharey=True)
plt.plot(dfitlr9['S1 Discharge'])
plt.plot(vf_dis1['discharge']*0.02831 )
plt.show()
fig, axes = plt.subplots(1, 1, figsize=(10, 5), sharey=True)
plt.plot(dfitlr9['S2 Depth'])
plt.plot(vf_dep1['depth']*.3048)
plt.show()

#%%
