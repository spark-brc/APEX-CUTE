import parm, os
import msgbox
import main_prog


# identify which obsevered data files to use
def read():   
    import readobscrop

    filelist = []   
    fn = []
    parm.obs_val = [0 for x in range(len(parm.apex_outlets))]
    parm.obs_date = [0 for x in range(len(parm.apex_outlets))]
    #parm.nodata = [0 for x in range(len(parm.apex_outlets))]
    parm.start_obs = [0 for x in range(len(parm.apex_outlets))]
    parm.end_obs = [0 for x in range(len(parm.apex_outlets))]

    #Verify if obs files exist
    for file in os.listdir(parm.path_obs):
        if file.endswith(".csv"):
            filelist.append(file)

    for i in range(len(parm.apex_var)):
        fn = 'rch_'
        if parm.apex_var[i]<=8:
            fn = 'rch_'
            readfn(i,fn,filelist)
        elif parm.apex_var[i]<=23: 
            fn = 'sub_' 
            readfn(i,fn,filelist)
        elif parm.apex_var[i]<=27: 
            fn = 'crp_' 
            readfn(i,fn,filelist)
        elif parm.apex_var[i]==28: 
            readobscrop.read(i) #Read grain yield data, if any.
            if parm.iflg==1: return
        else:
            fn = 'lwe_' 
            readfn(i,fn,filelist)


        #Start/ending date
    for i in range(len(parm.apex_outlets)):
        if not (parm.apex_var[i]>=28 and parm.obs_yld_type=='9999'):
            parm.start_obs[i] = parm.obs_date[i][0]
            parm.end_obs[i] = parm.obs_date[i][len(parm.obs_date[i])-1]
 
            ##Days with no data
            #for io in range(len(parm.obs_date[i])):
            #    if parm.obs_val[i][io]<-99 or parm.obs_val[i][io]>=9999:
            #        parm.nodata += 1
        else:            
            parm.start_obs[i] = parm.start_pred[i]
            parm.end_obs[i] = parm.end_pred[i]

def readfn(i,fn,filelist):
    import readobsday
    import readobsmonth
    import readobsyear
    if parm.obs_dt[i].upper() == 'DAILY':
        filename = fn + 'daily' + str(parm.apex_outlets[i]) + '.csv'
        if not filename in filelist:
            parm.error_msg = filename + ' is not found'
            msgbox.msg("Error ", parm.error_msg)
            parm.iflg=1; return
        else:
            readobsday.read(i,fn)
    elif parm.obs_dt[i].upper() == 'MONTHLY':
        filename = fn + 'monthly' + str(parm.apex_outlets[i]) + '.csv'
        if not filename in filelist:
            parm.error_msg = filename + ' is not found'
            msgbox.msg("Error ", parm.error_msg)
            parm.iflg=1; return
        else:
            readobsmonth.read(i,fn)
    elif parm.obs_dt[i].upper() == 'YEARLY':
        filename = fn + 'yearly' + str(parm.apex_outlets[i]) + '.csv'
        if not filename in filelist:
            parm.error_msg = filename + ' is not found'
            msgbox.msg("Error ", parm.error_msg)
            parm.iflg=1; return
        else:
            readobsyear.read(i,fn)
