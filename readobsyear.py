import parm, datetime
from numpy import genfromtxt

def read(i,fn): #i: outlet ID#; fn: part of filename 
    #Reads yearly measured files 
    yr = 0

    #Config.dat Output Variables(1-9):Flow=1, Sed=2, TN=3, TP=4, MinN=5, OrgN=6, MinP=7, OrgP=8, TPest=9
    filen =parm.path_obs +'/'+ fn + 'yearly' + str(parm.apex_outlets[i]) + '.csv'
    obsdata = genfromtxt(filen,  delimiter=',', skip_header=1)

    if fn=='rch_':
        if parm.apex_var[i] == 0: # RCH-Flow
            parm.obs_val[i] = obsdata[:,1] 
        elif parm.apex_var[i] == 1: # RCH-Sediment
            parm.obs_val[i] = obsdata[:,2]
        elif parm.apex_var[i] == 2: # RCH-TN
            parm.obs_val[i] = obsdata[:,3]
        elif parm.apex_var[i] == 3: # RCH-TP
            parm.obs_val[i] = obsdata[:,4]
        elif parm.apex_var[i] == 4: # RCH-MineralN
            parm.obs_val[i] = obsdata[:,5]
        elif parm.apex_var[i] == 5: # RCH-OrganicN
            parm.obs_val[i] = obsdata[:,6]
        elif parm.apex_var[i] == 6: # RCH-MineralP
            parm.obs_val[i] = obsdata[:,7]
        elif parm.apex_var[i] == 7: # RCH-OrganicP
            parm.obs_val[i] = obsdata[:,8]
        elif parm.apex_var[i] == 8: # RCH-Pesticide
            parm.obs_val[i] = obsdata[:,9]
        
    if fn=='sub_':
        if parm.apex_var[i] == 9: # SUB-WaterYld
            parm.obs_val[i] = obsdata[:,1] 
        elif parm.apex_var[i] == 10: # SUB-Runoff
            parm.obs_val[i] = obsdata[:,2] 
        elif parm.apex_var[i] == 11: # SUB-QDr
            parm.obs_val[i] = obsdata[:,3] 
        elif parm.apex_var[i] == 12: # SUB-ResQ
            parm.obs_val[i] = obsdata[:,4] 
        elif parm.apex_var[i] == 13: # SUB-PET
            parm.obs_val[i] = obsdata[:,5] 
        elif parm.apex_var[i] == 14: # SUB-ET
            parm.obs_val[i] = obsdata[:,6] 
        elif parm.apex_var[i] == 15: # SUB-SW
            parm.obs_val[i] = obsdata[:,7] 
        elif parm.apex_var[i] == 16: # SUB-SedYLD
            parm.obs_val[i] = obsdata[:,8] 
        elif parm.apex_var[i] == 17: # SUB-QN
            parm.obs_val[i] = obsdata[:,9] 
        elif parm.apex_var[i] == 18: # SUB-QP
            parm.obs_val[i] = obsdata[:,10] 
        elif parm.apex_var[i] == 19: # SUB-ORGN
            parm.obs_val[i] = obsdata[:,11] 
        elif parm.apex_var[i] == 20: # SUB-ORGP
            parm.obs_val[i] = obsdata[:,12] 
        elif parm.apex_var[i] == 21: # SUB-QDRN
            parm.obs_val[i] = obsdata[:,13] 
        elif parm.apex_var[i] == 22: # SUB-QDRP
            parm.obs_val[i] = obsdata[:,14] 
        elif parm.apex_var[i] == 23: # SUB-STMP
            parm.obs_val[i] = obsdata[:,16] 

        if fn=='crp_':
            if parm.apex_var[i] == 24: # CRP-BIOM
                parm.obs_val[i] = obsdata[:,2] 
            elif parm.apex_var[i] == 25: # CRP-STL
                parm.obs_val[i] = obsdata[:,3] 
            elif parm.apex_var[i] == 26: # CRP-LAI
                parm.obs_val[i] = obsdata[:,4] 
            elif parm.apex_var[i] == 27: # CRP-CHT
                parm.obs_val[i] = obsdata[:,5] 

    if fn=='lwe_':
        if parm.apex_var[i] == 29: # SED-HFLUX g/m/d
            parm.obs_val[i] = obsdata[:,1] 
        elif parm.apex_var[i] == 30: # SED-VFLUX g/m/d
            parm.obs_val[i] = obsdata[:,2] 

    #Count the total number of days
    obs_row_count = obsdata.shape[0]
                
    #Calendar dates for observed data
    obsdate = [0 for x in range(obs_row_count)] 
    for j in range(obs_row_count):
        yr = int(float(obsdata[j,0]))
        obsdate[j] = datetime.date(yr, 1, 1) 
        if j>0:
            delta = obsdate[j].year - obsdate[j-1].year
            if delta>1:
                txt = 'Observation data is errornous between ' + str(obsdate[j-1]) + ' and ' + str(obsdate[j])
                parm.error_handle(txt)

    parm.obs_date[i] = obsdate[:]
    a=1
        
