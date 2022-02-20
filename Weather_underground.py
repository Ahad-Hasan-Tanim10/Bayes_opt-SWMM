# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 11:29:30 2021

@author: ATANIM
"""
import pandas as pd
import glob
import matplotlib.pyplot as plt

path = r'C:/Users/ATANIM/Documents/PhD research 2/Data/Calibration/Tide'
all_files = glob.glob(path + "/*.csv")
all_files

df_from_each_file = (pd.read_csv(f) for f in all_files)
df_merged = (pd.read_csv(f, sep=',') for f in all_files)
df_merged   = pd.concat(df_from_each_file, ignore_index=True)
df_merged.to_csv( "merged.csv")



#%%
fig, axs = plt.subplots(figsize=(12, 4))
df_merged['dm'] = pd.to_datetime(df_merged['Date Time']).dt.to_period('D')
dfd= df_merged.set_index('dm')
dfd1= dfd.resample('M').agg(['min','max'])
dfd1.to_csv(r'E:\CC\csv\tide.csv', index = True)

#%%
prec = pd.read_csv(r'E:\CC\NCEP\Charleston\Local climate data\Charleston_prec.csv', parse_dates = ['DATE'], index_col = 0)
prec1 = prec.resample('M').agg(['sum','mean','max'])
prec1.to_csv(r'E:\CC\csv\prec.csv', index = True)


#%%
wind = pd.read_csv(r'E:\CC\NCEP\Charleston\Local climate data\gldas wind speed\wind_84.csv', parse_dates = ['Date'], index_col = 0)
wind1 = wind.resample('M').agg(['mean'])
wind1.to_csv(r'E:\CC\NCEP\Charleston\Local climate data\gldas wind speed\wind1.csv', index = True)
#%%
import urllib.request
import json
#%%
def url1(year, month,day):
    if len(str(month)) == 1:
        month1 = str(0)+ str(month)
    elif len(str(month)) == 2:
        month1 = str(month)
    if len(str(day)) == 1:
        day1 = str(0)+ str(day)
    elif len(str(day)) == 2:
        day1 = str(day)
    url = r"https://api.weather.com/v2/pws/history/all?stationId={}&format=json&units=m&date={}{}{}&apiKey={}".format("KSCCHARL1179", str(year), month1, day1, "c63d7117618e48e0bd7117618ed8e019")
    data1 = urllib.request.urlopen(url)
    data2 = json.load(data1)
    l= data2["observations"]
    return l
#%%
from itertools import repeat
year = list(repeat(2021,3))
month = list(repeat(11,3))
day = list(range(5,8,1))
#%%
data = []
for i,j, k in zip(year, month,day):
    l= url1(i, j,k)
    data.extend(l)

#list = []
#for x in data:
#    list.append(x)
#%% not important. test run
import pandas as pd
date = []
rf = pd.DataFrame.from_dict(data, orient='columns')
#data['observations'][0] open a list
for precipTotal in data['observations']:
    print(precipTotal)
#access in to the precipitation data
data['observations'][0]['metric']['precipTotal']
data2 = data["observations"]
#%%
prec = []
time = []
for dic_ in data:
    prec.append(dic_["metric"]["precipTotal"])
    time.append(dic_["obsTimeLocal"])
#%%
time = pd.DataFrame(time)
rf =pd.DataFrame(prec)
ts = pd.merge(time, rf, left_index=True, right_index=True)
ts.columns = ['Time', 'rain']
#ts = ts.set_index(['Time'])
ts.rain.plot()
#%%
ts['TimeStamp'] = pd.to_datetime(ts['Time'], format='%Y-%m-%d %H:%M:%S')
ts['DayMax'] = ts.groupby(ts['TimeStamp'].dt.date)['rain'].transform('max')
#%%
ts['rr'] = ts['rain'].diff()
#%%
for index, row in n.iterrows():
    if row['rf'] == 1:
        row['df'] =1
    else:
        row['df']=5
    
#%%
for i in range(3):
    print(n.iloc[i][0])
#%%

gr2 = pd.DataFrame(ts['rain'])
gr2.columns = ['Rain']
gr2['dif'] = gr2['Rain'].diff()
gr2= []
#gr['dif'] = 0
for i in range(875):
    if gr2.iloc[i+1][1] -  gr2.iloc[i][1] < 0:
        gr2.iloc[i][1] = gr2.iloc[i][0]
    else:
        gr2.iloc[i][1] = 0
#%%
for i in range(len(ts)):
    if ts.loc[i+1,"rain"]-ts.loc[i,"rain"] < 0:
        ts.loc[i,"diff"]=ts.loc[i,"rain"]
    else:
        ts.loc[i,"diff"] = 0
#%%
ts.loc[875,"diff"] = 0
ts['diff1'] = ts['diff'].shift(+1)
ts.loc[0,"diff1"] = 0
ts['cs'] = ts['diff1'].cumsum(axis=0)
ts['cumulative_rf'] = ts['cs']+ ts['rain']
ts.cumulative_rf.plot()
ts = ts.set_index(['rain'])
rainfall2 = ts[['cumulative_rf','Time']].copy()
rainfall2.columns = [ 'rain', 'Time']
rainfall2['TimeStamp'] = pd.to_datetime(rainfall2['Time'], format='%Y-%m-%d %H:%M:%S')
rf = rainfall2.set_index('TimeStamp')
rainfall= rf.resample('5min').mean()
rainfall.to_csv(r"C:\Users\ATANIM\Documents\PhD research 2\icpr\Rainfall data\Flood2021\nov.csv", index =True)
