import parm, datetime
from numpy import genfromtxt
import msgbox

def read(i,fn):
    #Reads daily measured files and selects requiered data
    yr,mon,day = 0,0,0

    filename = '/' + fn + 'daily' + str(parm.apex_outlets[i]) + '.csv'
    obsdata = genfromtxt(parm.path_obs + filename,  delimiter=',', skip_header=1)   

    if fn=='rch_':
        if parm.apex_var[i] == 0: # RCH-Flow
            parm.obs_val[i] = obsdata[:,3] 
        elif parm.apex_var[i] == 1: # RCH-Sediment
            parm.obs_val[i] = obsdata[:,4]
        elif parm.apex_var[i] == 2: # RCH-TN
            parm.obs_val[i] = obsdata[:,5]
        elif parm.apex_var[i] == 3: # RCH-TP
            parm.obs_val[i] = obsdata[:,6]
        elif parm.apex_var[i] == 4: # RCH-MineralN
            parm.obs_val[i] = obsdata[:,7]
        elif parm.apex_var[i] == 5: # RCH-OrganicN
            parm.obs_val[i] = obsdata[:,8]
        elif parm.apex_var[i] == 6: # RCH-MineralP
            parm.obs_val[i] = obsdata[:,9]
        elif parm.apex_var[i] == 7: # RCH-OrganicP
            parm.obs_val[i] = obsdata[:,10]
        elif parm.apex_var[i] == 8: # RCH-Pesticide
            parm.obs_val[i] = obsdata[:,11]
        
    if fn=='sub_':
        if parm.apex_var[i] == 9: # SUB-WaterYld
            parm.obs_val[i] = obsdata[:,3] 
        elif parm.apex_var[i] == 10: # SUB-Runoff
            parm.obs_val[i] = obsdata[:,4] 
        elif parm.apex_var[i] == 11: # SUB-QDr
            parm.obs_val[i] = obsdata[:,5] 
        elif parm.apex_var[i] == 12: # SUB-ResQ
            parm.obs_val[i] = obsdata[:,6] 
        elif parm.apex_var[i] == 13: # SUB-PET
            parm.obs_val[i] = obsdata[:,7] 
        elif parm.apex_var[i] == 14: # SUB-ET
            parm.obs_val[i] = obsdata[:,8] 
        elif parm.apex_var[i] == 15: # SUB-SW
            parm.obs_val[i] = obsdata[:,9] 
        elif parm.apex_var[i] == 16: # SUB-SedYLD
            parm.obs_val[i] = obsdata[:,10] 
        elif parm.apex_var[i] == 17: # SUB-QN
            parm.obs_val[i] = obsdata[:,11] 
        elif parm.apex_var[i] == 18: # SUB-QP
            parm.obs_val[i] = obsdata[:,12] 
        elif parm.apex_var[i] == 19: # SUB-ORGN
            parm.obs_val[i] = obsdata[:,13] 
        elif parm.apex_var[i] == 20: # SUB-ORGP
            parm.obs_val[i] = obsdata[:,14] 
        elif parm.apex_var[i] == 21: # SUB-QDRN
            parm.obs_val[i] = obsdata[:,15] 
        elif parm.apex_var[i] == 22: # SUB-QDRP
            parm.obs_val[i] = obsdata[:,16] 
        elif parm.apex_var[i] == 23: # SUB-STMP
            parm.obs_val[i] = obsdata[:,18] 

    if fn=='crp_':
        if parm.apex_var[i] == 24: # CRP-BIOM
            parm.obs_val[i] = obsdata[:,4] 
        elif parm.apex_var[i] == 25: # CRP-STL
            parm.obs_val[i] = obsdata[:,5] 
        elif parm.apex_var[i] == 26: # CRP-LAI
            parm.obs_val[i] = obsdata[:,6] 
        elif parm.apex_var[i] == 27: # CRP-CHT
            parm.obs_val[i] = obsdata[:,7] 
 
    if fn=='lwe_':
        if parm.apex_var[i] == 29: # SED-HFLUX g/m/d
            parm.obs_val[i] = obsdata[:,3] 
        elif parm.apex_var[i] == 30: # SED-VFLUX g/m/d
            parm.obs_val[i] = obsdata[:,4] 


    #Count the total number of days
    obs_row_count = obsdata.shape[0]
                
    #Calendar dates for observed data
    obsdays = [0 for x in range(obs_row_count)] 
    for j in range(obs_row_count):
        yr = int(float(obsdata[j,0]))
        mon = int(float(obsdata[j,1]))
        day = int(float(obsdata[j,2]))
        obsdays[j] = datetime.date(yr, mon, day) 
        #Check Obs file for error
        if j>0:
            delta = obsdays[j] - obsdays[j-1]
            if delta.days>1:
                msg = 'Error is found in the obs data between ' + str(obsdays[j-1]) + ' and ' + str(obsdays[j])
                msgbox.msg("Message", msg)
                return
    
    parm.obs_date[i] = obsdays[:]