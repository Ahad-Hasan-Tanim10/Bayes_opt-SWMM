# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 10:05:20 2022

@author: ATANIM
"""

import os
import runpy

os.chdir(r'C:\Users\ATANIM\Documents\Research\Rocky branch\Flood')
#%%
def runy(runfile):
  with open(runfile,"r") as rnf:
    exec(rnf.read())
#%%
from datetime import datetime, timedelta
rdi1 =  "8/24/2022 16:30"
starttime_st = datetime.strptime(rdi1, "%m/%d/%Y %H:%M")
ST_time = []
ST_time.append('"'+ rdi1 + '"')
for l in range(0,10):
    starttime_f = starttime_st + timedelta(hours=3*l, minutes=00, seconds=00)
    startime_st = '"'+ starttime_f.strftime("%m/%d/%Y %H:%M")+ '"'
    startime_st1 = startime_st[:1] +  startime_st[2:]
    ST_time.append(startime_st1)

stoptime_start = datetime.strptime("8/24/2022 19:30", "%m/%d/%Y %H:%M")
STop_time = []
STop_time.append('"'+"8/24/2022 19:30"+ '"')
for s in range(0,9):
    stoptime_f = stoptime_start  + timedelta(hours=3*s, minutes=00, seconds=00)
    stoptime_st = '"'+ stoptime_f.strftime("%m/%d/%Y %H:%M") + '"'
    stoptime_st1 = stoptime_st[:1] +  stoptime_st[2:]
    STop_time.append(stoptime_st1)
stoptime_end = '"'+ "8/25/2022 22:00"+ '"'
STop_time.append(stoptime_end)

for id in range(0,10):
    print(STop_time[id], STop_time[id+1])
    print(ST_time[id], ST_time[id+1])
    
dfitlr =[]
dfitlr.append('dfitlr'+str(0))
for i in range(0,len(ST_time)-1):
    new_difltr = 'dfitlr'+str(i)
    dfitlr.append(new_difltr)
plist =[]
plist.append('plist'+str(0))
for i in range(0,len(ST_time)-1):
    new_plist = 'plist'+str(i)
    plist.append(new_plist)
#%%
import re

def replace(file, pattern, subst):
    # Read contents from file as a single string
    file_handle = open(file, 'r')
    file_string = file_handle.read()
    file_handle.close()

    # Use RE package to allow for replacement (also allowing for (multiline) REGEX)
    file_string = (re.sub(pattern, subst, file_string))

    # Write contents to file.
    # Using mode 'w' truncates the file.
    file_handle = open(file, 'w')
    file_handle.write(file_string)
    file_handle.close()

#%%
#================ reset the string to be changed in the py file========== 
#L177, L178, L232 replace them
replace("RTC_cal1.py", "8/25/2022 15:30", "8/25/2022 22:00")#update the stop report time
replace("RTC_cal1.py", "8/25/2022 12:30", "8/25/2022 15:30")#check the file first
#check the file first
replace("RTC_cal1.py", "dfitlr5", "dfitlr5")#check the file first
#runy("RTC_cal.py")
runpy.run_path(path_name='RTC_cal1.py')

#%%
for i,j,k,pl in zip(range(0,len(STop_time)-1), range(0,len(ST_time)-1), range(0,len(dfitlr)-1), range(0,len(plist)-1)):
    replace("RTC_cal1.py", STop_time[i],  STop_time[i+1])
    replace("RTC_cal1.py", ST_time[j],  ST_time[j+1])
    replace("RTC_cal1.py", dfitlr[k], dfitlr[k+1])
    replace("RTC_cal1.py", plist[pl], plist[pl+1])
    runpy.run_path(path_name='RTC_cal1.py')
#replace("RTC_cal1.py", STop_time[i],  STop_time[i+1])   
#%%
replace("RTC_cal.py", "8/25/2022 5:30", "8/25/2022 5:30")
replace("RTC_cal.py", "8/25/2022 10:30", "8/25/2022 11:00")