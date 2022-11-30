import os, parm, shutil
import numpy as np

def update(IYR):
    #This module updates APEX's DLY file through rainfall input error model
     
    #Read DLY files
    file_list = []
    fp = parm.path_proj + '\TxtInOut'
    for file in [doc for doc in os.listdir(fp) if doc.endswith('.dly')]:
        file_list.append(file)

    j = len(parm.par_name)
    #Laten_mu = parm.cur_test_var[j-2]
    #Laten_sigma = 0.01*parm.cur_test_var[j-1]    
    Laten_mu = parm.par_initval[j-2]
    Laten_sigma = 0.01*parm.par_initval[j-1]

    for i in file_list:
        fnam = fp + '/' + i
        fnamW = parm.path_TxtWork + '/' + i
        #shutil.copyfile(fnam,fnamW)
        parm.fnam1 = open(fnam, 'r')
    
        #Creat a temporary file
        try:
            parm.fnam2 = open(parm.path_TxtWork + '/' +'temp.dly','w') #temp file
        except:
            #Print error message and exit
            parm.error_handle('Error on creating temp.tmp for DLY file\n')

        lnum = 0
        for txtline in parm.fnam1:
            lnum += 1             
            if lnum ==1:            
                wid = len(txtline)

            if int(txtline[0:6]) >= IYR:
                t1 = txtline[0:33]
                t2 = txtline[33:39] # prep
                pcp_temp1 = float(t2)
                pcp_new1 = pcp_temp1*np.random.normal(Laten_mu,Laten_sigma)  # if prep=0, then still = 0 after the input error model
                txt = str("{:6.2f}".format(pcp_new1))    
                txtline1 = t1 + txt
                if wid < 41:
                   txtline = txtline1 + '\n'                       
                else:
                   t3 = txtline[39:len(txtline)]                    
                   txtline = txtline1 + t3   
                parm.fnam2.writelines(txtline)      
            
        parm.fnam1.close()
        parm.fnam2.close()
        os.remove(fnamW)
        os.rename(parm.path_TxtWork + '\temp.dly',fnamW)  

 