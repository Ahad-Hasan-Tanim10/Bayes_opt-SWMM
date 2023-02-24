# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 12:41:56 2022

@author: ATANIM
"""

import urllib.request
import json
def url1(year, month,day, stn):
    if len(str(month)) == 1:
        month1 = str(0)+ str(month)
    elif len(str(month)) == 2:
        month1 = str(month)
    if len(str(day)) == 1:
        day1 = str(0)+ str(day)
    elif len(str(day)) == 2:
        day1 = str(day)
    url = r"https://api.weather.com/v2/pws/history/all?stationId={}&format=json&units=m&date={}{}{}&apiKey={}".format(stn, str(year), month1, day1, "c63d7117618e48e0bd7117618ed8e019")
    data1 = urllib.request.urlopen(url)
    data2 = json.load(data1)
    l= data2["observations"]
    return l

#%%

#%%
# not faster approcah 
def sub(x,a):
    inp[sections.SUBCATCHMENTS][x].Width = a 
    return a
#%%
#%%
from scipy import stats
list_of_dists = ['alpha','anglit','arcsine','beta','betaprime','bradford','burr','burr12','cauchy','chi','chi2','cosine','dgamma','dweibull','erlang','expon','exponnorm','exponweib','exponpow','f','fatiguelife','fisk','foldcauchy','foldnorm','genlogistic','genpareto','gennorm','genexpon','genextreme','gausshyper','gamma','gengamma','genhalflogistic','gilbrat','gompertz','gumbel_r','gumbel_l','halfcauchy','halflogistic','halfnorm','halfgennorm','hypsecant','invgamma','invgauss','invweibull','johnsonsb','johnsonsu','kstwobign','laplace','levy','levy_l','logistic','loggamma','loglaplace','lognorm','lomax','maxwell','mielke','nakagami','ncx2','ncf','nct','norm','pareto','pearson3','powerlaw','powerlognorm','powernorm','rdist','reciprocal','rayleigh','rice','recipinvgauss','semicircular','t','triang','truncexpon','truncnorm','tukeylambda','uniform','vonmises','vonmises_line','wald','weibull_min','weibull_max']
#%capture --no-stdout

#%%

def fitdist(colx):
    val = print("result_",colx)
    val = list()
    for i in list_of_dists:
        dist = getattr(stats, i)
        param = dist.fit(colx)
        ald = stats.kstest(colx, i, args=param)
        val.append((i,ald[0],ald[1]))# ald[0] is the KS test result and ald[1] is the p-value
    val.sort(key=lambda x:float(x[2]), reverse=True)# sort your result based on p value
    return val
#%%