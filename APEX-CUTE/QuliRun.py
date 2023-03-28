import numpy as np
import parm

def save():  
  #  RunID, VarID, re_c,nse_c,re_v,nse_v = np.loadtxt('modPerf.out', skiprows=1, usecols=(0,2,3,5,10,12), unpack=True) #skip first title row 
    fname = parm.path_proj + '\\modPerf.out'
    fmod = open(fname,'r')
    
    fout = parm.path_proj + '\\QuliRun.out'
    fnam = open(fout,'w') # Qualified runs

    lnum = 0
    for txtline in fmod:
        t = txtline.split()
        lnum += 1
        if lnum >1:
           try:
                t1 = int(t[2])
           except:
                t1=99999
           if t1 == 1: # flow               
               if abs(float(t[3])) < 20 and float(t[5]) > 0.5: # calibration period       
                   if len(t) > 10:
                       if abs(float(t[10])) < 20 and float(t[12]) > 0.5: # validation period       
                            fnam.writelines(txtline) 
                   else: # no validation period 
                       fnam.writelines(txtline)        
           if t1 == 2: # sediment               
               if abs(float(t[3])) < 35 and float(t[5]) > 0.45: 
                    if len(t) > 10:      
                        if abs(float(t[10])) < 40 and float(t[12]) > 0.45:	       
                           fnam.writelines(txtline) 
                    else:
                        fnam.writelines(txtline) 
        else:
             fnam.writelines(txtline)
    fnam.close()
    fmod.close()
            