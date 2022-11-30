import parm, math, datetime
import numpy as np
import msgbox

def select_file():
    #Read APEX output values from SAD file
    for i in range(0,len(parm.apex_outlets)):
        if parm.apex_output[i]=='LWE':
            LWEOUTread()
            break
    
def LWEOUTread():
    #This module reads APEX output from *.DWS file.  
    
    fnam = parm.path_TxtWork + '/Landscape wind erosion.out'
    try:
        lwe_data = np.genfromtxt(fnam,delimiter='',skip_header=1) #lwe_data[row,column]
    except:
        #Print error message and exit
        parm.error_msg = fnam + ' is not found.'
        msgbox.msg("Error", parm.error_msg)
        parm.iflg=1; return

    var1 = [0 for x in range(len(lwe_data))]        
    icount = 0

    for i in range(0,len(parm.apex_var)):
        if parm.apex_var[i]==29:             #Horizontal flux
            var1 = lwe_data[:,5]
        elif parm.apex_var[i]==30:             #Vertical flux
            var1 = lwe_data[:,6]

    #Aggregate output 
        ndata = len(parm.pred_date[i])                
        parm.pred_datea[i] = parm.pred_date[i]
        ival = [0 for x in range(ndata)]
        idt = 0
 
        if parm.obs_dt[i].upper() == 'DAILY':
            ival = var1
        elif parm.obs_dt[i].upper() == 'MONTHLY':     
            for j in range(len(var1)-1):
                jday = datetime.date(int(lwe_data[j,0]),int(lwe_data[j,1]),int(lwe_data[j,2]))
                jday1 = datetime.date(int(lwe_data[j+1,0]),int(lwe_data[j+1,1]),int(lwe_data[j+1,2]))
                ival[idt] = ival[idt] + var1[j]
                if jday.month != jday1.month:
                    idt += 1

        elif parm.obs_dt[i].upper() == 'YEARLY':     
             for j in range(len(var1)-1):
                jday = datetime.date(int(lwe_data[j,0]),int(lwe_data[j,1]),int(lwe_data[j,2]))
                jday1 = datetime.date(int(lwe_data[j+1,0]),int(lwe_data[j+1,1]),int(lwe_data[j+1,2]))
                ival[idt] = ival[idt] + var1[j]
                if jday.year != jday1.year:
                    idt += 1


        parm.pred_val[i] = ival
