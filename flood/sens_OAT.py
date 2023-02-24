# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 13:18:29 2023

@author: ATANIM
"""


from swmm_api.input_file.section_labels import SUBCATCHMENTS,SUBAREAS, INFILTRATION, TRANSECTS
from swmm_api.input_file import read_inp_file, SwmmInput, section_labels as sections
import numpy as np
import pandas as pd
from pyswmm import Simulation, Subcatchments, Nodes, Links

inp = read_inp_file(r'C:\Users\ATANIM\Documents\Research\Rocky branch\Flood\Calibration_m.inp', convert_sections=[SUBCATCHMENTS,SUBAREAS, INFILTRATION, TRANSECTS])  # type: swmm_api.SwmmInput


#%%
def sub(x,a):
    inp[sections.SUBCATCHMENTS][x].Width = a 
    return a
#%%
def sl(x,a):
    inp[sections.SUBCATCHMENTS][x].Slope = a 
    return a
#%%
def cn(x,a):
    inp[sections.INFILTRATION][x].Psi = a 
    return a
#%%
def Imperv(x,a):
    inp[sections.SUBCATCHMENTS][x].Imperv = a 
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
def ZeroImp(x,a):
    inp[sections.SUBAREAS][x].PctZero = a
    return a
#%%
def SImperv(x,a):
    inp[sections.SUBAREAS][x].S_Imperv = a
    return a
#%%
def Sperv(x,a):
    inp[sections.SUBAREAS][x].S_Perv = a
    return a
#%%
catchment = ['ws-128', 'ws-148', 'ws-159', 'ws-71', 'ws-84', 'ws-161', 'ws-17']  
sw = []
slc = []
sN = []
Impervious = []
Nimpl = []
Npervl = []
Pct0 = []
Simpl = []
Sprl = []
for i in catchment:
    rd = inp[sections.SUBCATCHMENTS][i].Width
    sw.append(rd)
    cd = inp[sections.INFILTRATION][i].Psi
    sN.append(cd)
    imp_percentage = inp[sections.SUBCATCHMENTS][i].Imperv
    Impervious.append(imp_percentage)
    slci = inp[sections.SUBCATCHMENTS][i].Slope
    slc.append(slci)
    im = inp[sections.SUBAREAS][i].N_Imperv
    Nimpl.append(im)
    per = inp[sections.SUBAREAS][i].N_Perv
    Npervl.append(per)
    zeropct = inp[sections.SUBAREAS][i].PctZero
    Pct0.append(zeropct)
    Sim = inp[sections.SUBAREAS][i].S_Imperv
    Simpl.append(Sim)
    Spr = inp[sections.SUBAREAS][i].S_Perv
    Sprl.append(Spr)
    print(Sprl)
    print(Simpl)
    print(sw)
    print(slc)
#%%
import os
def runf_sens(fn):
    f_path = r'C:\Users\ATANIM\Documents\Research\Rocky branch\Flood'
    sim_path = os.path.join(f_path, "%s.inp"% fn)
    with Simulation(sim_path) as sim:
        S1 = Subcatchments(sim)["ws-84"] 
    # we need to create a time series table, idx is the time, s1_values, s2_values are the two columns for the runoff of s1 and s1
        idx = []
        s1_values = []
    # as we are looping through each time step, we add the simulated value into the 3 coloumns variables above
        for step in sim:
            idx.append(sim.current_time)
            s1_values.append(S1.runoff)
    # using this line below, we turn the 3 columns into a table, called DataFrame using the pandas library. So that we can plot it.
        df = pd.DataFrame({'S1 Runoff': s1_values}, index=idx)
        return df
    
#%%
def inpfi(ftype, m3, n3, fn):
    if ftype == "sw":
        sub(m3,n3)
    elif ftype == "slope":
        sl(m3,n3)
    elif ftype == "imp":
        Imperv(m3,n3)
    elif ftype == "curve":
        cn(m3,n3)
    elif ftype == "Imperv_N":
        Nimp(m3,n3)
    elif ftype == "Perv_N":
        Nperv(m3,n3)
    elif ftype == "Imperv_S":
        SImperv(m3,n3)
    elif ftype == "Perv_S":
        Sperv(m3,n3)
    elif ftype == "0pct":
        ZeroImp(m3,n3)
    f_path = r'C:\Users\ATANIM\Documents\Research\Rocky branch\Flood'
    return inp.write_file(os.path.join(f_path, "%s.inp"% fn))

#%%
def sensti(ftype, w_name, par):
    inpfi(ftype, w_name, par, "Calibration_m1")
    di = runf_sens("Calibration_m1")
    return max(di['S1 Runoff'])



#%% create uniform timesteps for iteration
def mstep(upper,lower):
    return (upper-lower)/20
#%%
sub_width = []
for i in np.arange(0.5*sw[4], 1.55*sw[4],mstep(1.55*sw[4],0.5*sw[4])):
    pio = i
    sub_width.append(pio)
peak_fl1 = []
for j in sub_width:
    pfi1 = sensti("sw", "ws-84", j)
    peak_fl1.append(pfi1)

#%%
sub_slope = []
for slo in np.arange(0.5*slc[4], 1.55*slc[4],mstep(1.55*slc[4], 0.5*slc[4])):
    sub_slope.append(slo)
    
peak_fl2 = []
for slope_des in sub_slope:
    pfi2 = sensti("slope", "ws-84", slope_des)
    peak_fl2.append(pfi2)

#%%
sub_pct = []
for pct in np.arange(0.75*Pct0[4], 1.30*Pct0[4],mstep(1.30*Pct0[4], 0.75*Pct0[4])):
    sub_pct.append(pct)
    
peak_fl3 = []
for pct_des in sub_pct:
    pfi3 = sensti("0pct", "ws-84", pct_des)
    peak_fl3.append(pfi3)
#%%
sub_curve = []
for cnu in np.arange(0.75*sN[5], 1.30*sN[5],mstep(1.30*sN[5], 0.75*sN[5])):
    sub_curve.append(cnu)
    
peak_fl4 = []
for cn_des in sub_curve:
    pfi4 = sensti("curve", "ws-84", cn_des)
    peak_fl4.append(pfi4)
#%%
#%%
sub_imp = []
for impr in np.arange(0.5*Impervious[4], 1.55*Impervious[4],mstep(1.55*Impervious[4], 0.5*Impervious[4])):
    sub_imp.append(impr)
    
peak_fl5 = []
for imp_des in sub_imp:
    pfi5 = sensti("imp", "ws-84", imp_des)
    peak_fl5.append(pfi5)
#%%
sub_nimperv = []
for nimpr in np.arange(0.75*Nimpl[4], 1.30*Nimpl[4], mstep(1.30*Nimpl[4],0.75*Nimpl[4])):
    sub_nimperv.append(nimpr)

peak_fl6 = []
for nimp_des in sub_nimperv:
    pfi6 = sensti("Imperv_N", "ws-84", nimp_des)
    peak_fl6.append(pfi6)
    
#%%
sub_nperv = []
for nper in np.arange(0.75*Npervl[4], 1.30*Npervl[4],mstep(1.30*Npervl[4], 0.75*Npervl[4])):
    sub_nperv.append(nper)

peak_fl7 = []
for nper_des in sub_nperv:
    pfi7 = sensti("Imperv_N", "ws-84", nper_des)
    peak_fl7.append(pfi7)
#%%
sub_Sperv = []
for sper in np.arange(0.25*Sprl[4], 1.80*Sprl[4],mstep(1.80*Sprl[4], 0.25*Sprl[4])):
    sub_Sperv.append(sper)

peak_fl8 = []
for sper_des in sub_Sperv:
    pfi8 = sensti("Perv_S", "ws-84", sper_des)
    peak_fl8.append(pfi8)
#%%
sub_Simperv = []
for simper in np.arange(0.5*Simpl[4], 1.55*Simpl[4],mstep( 1.55*Simpl[4], 0.5*Simpl[4])):
    sub_Simperv.append(simper)

peak_fl9 = []
for simper_des in sub_Simperv:
    pfi9 = sensti("Imperv_S", "ws-84", simper_des)
    peak_fl9.append(pfi9)
#%%