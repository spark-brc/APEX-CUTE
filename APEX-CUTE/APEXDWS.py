import parm, math, datetime
import numpy as np
import msgbox

def select_file():
    #Read APEX output values from SAD file
    for i in range(len(parm.cs_name)):
        if parm.cs_name[i].lower()=='rto_bf' or parm.cs_name[i].lower()=='pet':
            DWSread()
            break
    
def DWSread():
    #This module reads APEX output from *.DWS file.  
    
    fnam = parm.path_TxtWork + '/' + parm.APEXRun_name + '.DWS'
    try:
        dws_data = np.genfromtxt(fnam,delimiter='',skip_header=9) #dws_data[row,column]
    except:
        #Print error message and exit
        parm.error_msg = fnam + ' is not found.'
        msgbox.msg("Error", parm.error_msg)
        parm.iflg=1; return

    #Baseflow ratio
    parm.baseflow_ratio = max(0., 1.- sum(dws_data[:,5]) / sum(dws_data[:,4])) #baseflow/Water_yield
    decimal_years = (parm.end_pred[0] - parm.start_pred[0]).days / 365.25
    parm.pet = sum(dws_data[:,8]) / decimal_years #annual PET in mm
    #var1 = [0 for x in range(len(dws_data))]        
    #icount = 0

    #for i in range(0,len(parm.apex_var)):
    #    if parm.apex_var[i]==0:             #surface runoff,mm
    #        var1 = dws_data[:,15]
    #    elif parm.apex_var[i]==1:             #flow,mm
    #        var1 = dws_data[:,4]
    #    elif parm.apex_var[i]==2:           #sediment,tons/ha
    #        var1 = dws_data[:,5] 
    #    elif parm.apex_var[i]==3:           #TN = QN+YN+QDRN+RSFN+QRFN,kg/ha
    #        var1 = dws_data[:,6] + dws_data[:,7] + dws_data[:,8] + dws_data[:,9] + dws_data[:,10] 
    #    elif parm.apex_var[i]==4:           #TP = QP+YP+QRFP+QDRP,kg/ha
    #        var1 = dws_data[:,11] + dws_data[:,12] + dws_data[:,13] + dws_data[:,14] 
    #    elif parm.apex_var[i]==5:           #Mineral N,kg/ha
    #        var1 = dws_data[:,6] + dws_data[:,8] + dws_data[:,9] + dws_data[:,10]
    #    elif parm.apex_var[i]==6:           #Organic N,kg/ha
    #        var1 = dws_data[:,7]
    #    elif parm.apex_var[i]==7:           #Mineral p,kg/ha
    #        var1 = dws_data[:,11] + dws_data[:,13] + dws_data[:,14]
    #    elif parm.apex_var[i]==8:           #Organic p,kg/ha
    #        var1[icount] = dws_data[:,12]
    #    #no pesticide data available in *.dws

    ##Aggregate output 
    #    ndata = len(parm.pred_date)                
    #    parm.pred_datea[i] = parm.pred_date 
    #    ival = [0 for x in range(ndata)]
    #    idt = 0
 
    #    if parm.obs_dt[i].upper() == 'D':
    #        ival = var1
    #    elif parm.obs_dt[i].upper() == 'M':     
    #        for j in range(len(var1)-1):
    #            jday = datetime.date(int(dws_data[j,0]),int(dws_data[j,1]),int(dws_data[j,2]))
    #            jday1 = datetime.date(int(dws_data[j+1,0]),int(dws_data[j+1,1]),int(dws_data[j+1,2]))
    #            ival[idt] = ival[idt] + var1[j]
    #            if jday.month != jday1.month:
    #                idt += 1

    #    elif parm.obs_dt[i].upper() == 'Y':     
    #         for j in range(len(var1)-1):
    #            jday = datetime.date(int(dws_data[j,0]),int(dws_data[j,1]),int(dws_data[j,2]))
    #            jday1 = datetime.date(int(dws_data[j+1,0]),int(dws_data[j+1,1]),int(dws_data[j+1,2]))
    #            ival[idt] = ival[idt] + var1[j]
    #            if jday.year != jday1.year:
    #                idt += 1


    #    parm.pred_val[i] = ival
