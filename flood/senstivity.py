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
from sklearn.metrics import r2_score

inp = read_inp_file(r'C:\Users\ATANIM\Documents\Research\Rocky branch\Flood\Calibration_fcm22.inp', convert_sections=[SUBCATCHMENTS,SUBAREAS, INFILTRATION, TRANSECTS])  # type: swmm_api.SwmmInput

#%%
dict = {"fruits": 56}
def change_val(x,val):
    dict[x]  = val
    return val
change_val("fruits",30)
dict
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
#%% imperviousness percentage of each catchment
def Sperv(x,a):
    inp[sections.SUBCATCHMENTS][x].Imperv = a 
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
    Spr = inp[sections.SUBCATCHMENTS][i].Imperv
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
vf_dis = pd.read_csv(r'C:\Users\ATANIM\Documents\Research\Rocky branch\Discharge data\discharge_22.csv')
#vf_dis = pd.read_csv(r'C:\Users\ATANIM\Documents\Research\Rocky branch\Validation\02169505CALdis.txt', sep='\t')
#vf_dis = pd.read_csv(r'C:\Users\ATANIM\Documents\Research\Rocky branch\Discharge data\discharge_21.csv')
vf_dis['Time'] =  pd.to_datetime(vf_dis['datetime'], format='%m/%d/%Y %H:%M')
vf_dis1 = vf_dis.set_index(['Time']).shift(0)
vf_dis1.pop('datetime')
#%%
def dis(mod, obs):
    dep = mod.resample("15min").mean().to_frame()
    d1 = dep["S1 Discharge"].to_numpy()
    d2 = ((obs["Discharge"]*0.027).to_numpy())#[:-2]
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
def inpfi(cnu, Nimplr, Nperlv, slcl, swl, Simplr, Spervl, n_chan, n_right, n_left, fn):
    for i,j in zip(catchment, cnu):
        cn(i,j)
    for e,f in zip(catchment, Nimplr):
        Nimp(e,f)
    for g,h in zip(catchment,Nperlv):
        Nperv(g,h)
    for (m2,n2) in zip (catchment,slcl):
        sl(m2,n2)
    for (m3,n3) in zip (catchment,swl):
        sub(m3,n3)
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
               a5, b5, c5, d5, e5, f5, g5,
               a6, b6, c6, d6, e6, f6, g6,
               a3, b3, c3, d3, e3, f3, g3,
               a4, b4, c4, d4, e4, f4, g4,
               i,j,k,l, h, m, n,o, p, q, r, s, fn):
    CNU = [cn('ws-128',(sN[0]+a)), cn('ws-148',(sN[1]+b)),cn('ws-159',(sN[2]+c)), cn('ws-71',(sN[3]+d)), cn('ws-84',(sN[4]+e)),
           cn('ws-161',(sN[5]+f)), cn('ws-17',(sN[6]+g))]
    n_imp = [Nimp('ws-128',(Nimpl[0]+a1)), Nimp('ws-148',(Nimpl[1]+b1)),Nimp('ws-159',(Nimpl[2]+c1)), Nimp('ws-71',(Nimpl[3]+d1)), Nimp('ws-84',(Nimpl[4]+e1)),
           Nimp('ws-161',(Nimpl[5]+f1)), Nimp('ws-17',(Nimpl[6]+g1))]
    n_per = [Nperv('ws-128',(Npervl[0]+a2)), Nperv('ws-148',(Npervl[1]+b2)),Nperv('ws-159',(Npervl[2]+c2)), Nperv('ws-71',(Npervl[3]+d2)), Nperv('ws-84',(Npervl[4]+e2)),
           Nperv('ws-161',(Npervl[5]+f2)), Nperv('ws-17',(Npervl[6]+g2))]
    slo = [sl('ws-128',(slc[0]+a5)), sl('ws-148',(slc[1]+b5)),sl('ws-159',(slc[2]+c5)), sl('ws-71',(slc[3]+d5)), sl('ws-84',(slc[4]+e5)),
           sl('ws-161',(slc[5]+f5)), sl('ws-17',(slc[6]+g5))]
    swo = [sub('ws-128',(sw[0]+a6)), sub('ws-148',(sw[1]+b6)),sub('ws-159',(sw[2]+c6)), sub('ws-71',(sw[3]+d6)), sub('ws-84',(sw[4]+e6)),
           sub('ws-161',(sw[5]+f6)), sub('ws-17',(sw[6]+g6))]    
    S_imper =[Simperv('ws-128',(Simpl[0]+a3)), Simperv('ws-148',(Simpl[1]+b3)),Simperv('ws-159',(Simpl[2]+c3)), Simperv('ws-71',(Simpl[3]+d3)), Simperv('ws-84',(Simpl[4]+e3)),
           Simperv('ws-161',(Simpl[5]+f3)), Simperv('ws-17',(Simpl[6]+g3))]
    S_per = [Sperv('ws-128',(Sprl[0]+a4)), Sperv('ws-148',(Sprl[1]+b4)),Sperv('ws-159',(Sprl[2]+c4)), Sperv('ws-71',(Sprl[3]+d4)), Sperv('ws-84',(Sprl[4]+e4)),
           Sperv('ws-161',(Sprl[5]+f4)), Sperv('ws-17',(Sprl[6]+g4))]
    n_ch = [manch('c67',(n_chan[0]+i)),(manch('c68',n_chan[1])+j), (manch('c70',n_chan[2])+k), (manch('c71',n_chan[3])+l)]
    n_r = [manchr('c67',(n_right[0]+h)), manchr('c68',(n_right[1]+m)), manchr('c70',(n_right[2]+n)), manchr('c71',(n_right[3]+o))]
    n_l = [manchl('c67',(n_left[0]+p)), manchl('c68',(n_left[1]+q)), manchl('c70',(n_left[2]+r)), manchl('c71',(n_left[3]+s))]
    inpfi(CNU,n_imp, n_per, slo, swo, S_imper,S_per, n_ch,n_r,n_l, fn)
    df = runf(fn)
    r2 = dis(df["S1 Discharge"], vf_dis1)
    return r2
#%%
def fr61(a, b, c, d, e, f, g,a1, b1, c1, d1, e1, f1, g1, a2, b2, c2, d2, e2, f2, g2,
         a5, b5, c5, d5, e5, f5, g5,a6, b6, c6, d6, e6, f6, g6,
         a3, b3, c3, d3, e3, f3, g3, a4, b4, c4, d4, e4, f4, g4,
         i,j,k,l, h, m, n,o,p, q, r, s):
    # print(params)  # <-- you'll see that params is a NumPy array
    # <-- for readability you may wish to assign names to the component variables
    r2 = modelswmmi(a, b, c, d,  e, f, g, a1, b1, c1, d1, e1, f1, g1,a2, b2, c2, d2, e2, f2, g2,
                    a5, b5, c5, d5, e5, f5, g5,a6, b6, c6, d6, e6, f6, g6,
                    a3, b3, c3, d3, e3, f3, g3, a4, b4, c4, d4, e4, f4, g4,
                    i,j,k,l, h, m, n,o,p, q, r, s, "Calibration_fcm22_TRI")
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

Z = []
e = []
e1 = []
e2 = []
e5 = []
e6 = []
e3 = []
e4 = []
ir = []
hr =[]
pr = []

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
              sherpa.Continuous(name='a2', range=[-0.005,0.6]),
              sherpa.Continuous(name='b2', range=[-0.005,0.6]),
              sherpa.Continuous(name='c2', range=[-0.005,0.6]),
              sherpa.Continuous(name='d2', range=[-0.005,0.4]),
              sherpa.Continuous(name='e2', range=[-0.005,0.6]),
              sherpa.Continuous(name='f2', range=[-0.005,0.6]),
              sherpa.Continuous(name='g2', range=[-0.005,0.6]),
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
              sherpa.Continuous(name='g6', range=[-0.3*sw[6],+0.7*sw[6]]),              
              sherpa.Continuous(name='a3', range=[-0.5,5.5]),
              sherpa.Continuous(name='b3', range=[-0.5,5.5]),
              sherpa.Continuous(name='c3', range=[-0.5,5.5]),
              sherpa.Continuous(name='d3', range=[-0.5,5.5]),
              sherpa.Continuous(name='e3', range=[-0.5,5.5]),
              sherpa.Continuous(name='f3', range=[-0.5,5.5]),
              sherpa.Continuous(name='g3', range=[-0.5,5.5]),
              sherpa.Continuous(name='a4', range=[-0.5*Sprl[0],0.5*Sprl[0]]),#imperviousness 
              sherpa.Continuous(name='b4', range=[-0.5*Sprl[1],0.5*Sprl[1]]),#imperviousness 
              sherpa.Continuous(name='c4', range=[-0.5*Sprl[2],0.5*Sprl[2]]),#imperviousness 
              sherpa.Continuous(name='d4', range=[-0.5*Sprl[3],0.5*Sprl[3]]),#imperviousness 
              sherpa.Continuous(name='e4', range=[-0.5*Sprl[4],0.5*Sprl[4]]),#imperviousness 
              sherpa.Continuous(name='f4', range=[-0.5*Sprl[5],0.5*Sprl[5]]),#imperviousness 
              sherpa.Continuous(name='g4', range=[-0.5*Sprl[6],0.5*Sprl[6]]),#imperviousness 
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
                                     max_concurrent=1,
                                     max_num_trials =100)
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

    num_iterations = 1
    for i in range(num_iterations):
        
        # access parameters via trial.parameters and id via trial.id
        pseudo_loss = fr61(trial.parameters['a'], trial.parameters['b'], 
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
                          trial.parameters['a5'],
                          trial.parameters['b5'], trial.parameters['c5'],
                          trial.parameters['d5'],trial.parameters['e5'],
                          trial.parameters['f5'],trial.parameters['g5'],
                           trial.parameters['a6'],
                          trial.parameters['b6'], trial.parameters['c6'],
                          trial.parameters['d6'],trial.parameters['e6'],
                          trial.parameters['f6'],trial.parameters['g6'],
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
    e.append(trial.parameters['e'])
    e1.append(trial.parameters['e1'])
    e2.append(trial.parameters['e2'])
    e5.append(trial.parameters['e5'])
    e6.append(trial.parameters['e6'])
    e3.append(trial.parameters['e3'])
    e4.append(trial.parameters['e4'])
    ir.append(trial.parameters['i'])
    hr.append(trial.parameters['h'])
    pr.append(trial.parameters['p'])
    print(pseudo_loss)

#%%

zipped = list(zip(e, e1, e2, e5, e6, e3, e4, ir, hr, pr))
distr = pd.DataFrame(zipped, columns=['e', 'e1', 'e2', 'e5', 'e6', 'e3', 'e4', 'ir', 'hr', 'pr'])
distr['e'] = sN[4]+ distr['e']
distr['e1'] = Nimpl[4]+ distr['e1']
distr['e2'] = Npervl[4]+ distr['e2']
distr['e5'] = slc[4]+ distr['e5']
distr['e6'] = sw[4]+ distr['e6']
distr['e3'] = Simpl[4]+ distr['e3']
distr['e4'] = Sprl[4]+ distr['e4']
distr['ir'] = n_chan[0]+ distr['ir']
distr['hr'] = n_right[0]+ distr['hr']
distr['pr'] = n_left[0]+ distr['pr']

#%%
import seaborn as sns
# set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above) 
sns.set(style="darkgrid")
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
fig.suptitle('Eventwise parameter distribution')
#sns.violinplot(ax=axes[0], data = distr['e'], linewidth=1)
sns.violinplot(ax=axes[0], data = distr['e3'], linewidth=1)
plt.show()
#%%

sns.set(font='Times New Roman')
sns.set_style("whitegrid")
sns.set(font_scale=2)
f, ax = plt.subplots(figsize=(4, 6), )
sns.violinplot(y=distr["e4"],color= 'blue')
ax.set_ylim(0, 100)
plt.ylabel('%Slope', fontsize=32);
#%%
excel_file = 'E:/Rocky Branch/Senstivity/Optimizers.xlsx'
all_sheets = pd.read_excel(excel_file, sheet_name=None)
#%%
sns.set(style="whitegrid")
sns.set(style="ticks")

# List of colors: https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
fig, axes = plt.subplots(5, 2, figsize=(14, 20), dpi = 300, sharey=False)
my_pal = ('darkorange', 'tan', 'deepskyblue', 'darkorchid', 'lawngreen', 'olivedrab')
axes = axes.flatten()
sns.set(font_scale=1)
sns.violinplot(ax=axes[0], data = all_sheets['Width'], linewidth=0.5, inner = 'quartiles', palette = my_pal)
sns.violinplot(ax=axes[1], data = all_sheets['CN'], linewidth=0.5, inner = 'quartiles', palette = my_pal)
sns.violinplot(ax=axes[2], data = all_sheets['Imperviouseness'],inner = 'quartiles', linewidth=0.5, palette = my_pal)
sns.violinplot(ax=axes[3], data = all_sheets['N-Perv'], linewidth=0.5, inner = 'quartiles', palette = my_pal)
sns.violinplot(ax=axes[4], data = all_sheets['N_Imperv'], linewidth=0.5, inner = 'quartiles', palette = my_pal)
sns.violinplot(ax=axes[5], data = all_sheets['%Slope'], linewidth=0.5, inner = 'quartiles', palette = my_pal)
sns.violinplot(ax=axes[6], data = all_sheets['N_left'], linewidth=0.5,inner = 'quartiles', palette = my_pal)
sns.violinplot(ax=axes[7], data = all_sheets['N_main'], linewidth=0.5,inner = 'quartiles', palette = my_pal)
sns.violinplot(ax=axes[8], data = all_sheets['N_right'], linewidth=0.5,inner = 'quartiles', palette = my_pal)


plt.grid(False)
plt.show()

#%%
dyfix = 'E:/Rocky Branch/Senstivity/Fixed_param.xlsx'
df_sheets = pd.read_excel(dyfix, sheet_name=None)
sns.set(style="whitegrid")
sns.set(style="ticks")


fig, axes = plt.subplots(2, 2, figsize=(12, 6), dpi = 300, sharey=False)
my_pal = ('blue', 'lightseagreen', 'olivedrab')
axes = axes.flatten()
sns.set(font_scale=1)
sns.violinplot(ax=axes[0], data = df_sheets['cn'], linewidth=0.5, inner = 'quartiles', palette = my_pal)
sns.violinplot(ax=axes[1], data = df_sheets['n_imp'], linewidth=0.5, inner = 'quartiles', palette = my_pal)
sns.violinplot(ax=axes[2], data = df_sheets['n_per'],inner = 'quartiles', linewidth=0.5, palette = my_pal)
sns.violinplot(ax=axes[3], data = df_sheets['swo'], linewidth=0.5, inner = 'quartiles', palette = my_pal)

plt.grid(False)
plt.show()