import os, parm
import numpy as np
import msgbox
import main_prog


# saved for soil update. will work on it
# Works only for the soil assigned for subarea 1 -Jaehak Jeong 6/17/2015

def update():
    #This module updates APEX's OPC file/s with new set of parameters
 
    #Read OPC file #only one file for now
    fnam = parm.path_TxtWork + '/' + parm.fnam_sol #error corrected: fnam_opc -> fnam_sol
    try:
        parm.fnam1 = open(fnam, 'r+')
    except:
        #Print error message and exit
        parm.error_msg = fnam + ' is not found.'
        msgbox.msg("Error", parm.error_msg)
        parm.iflg=1; return

    try:
         parm.fnam2 = open(parm.path_TxtWork + '/' + 'temp.tmp','w') #temp file
    except:
        #Print error message and exit
        parm.error_msg = "An error was occurred while \n creating temp.tmp for OPS file."
        msgbox.msg("Error", parm.error_msg)
        parm.iflg=1; return

    lnum = 1
    Line1 = np.zeros(10)

    for txtline in parm.fnam1:
        if lnum == 31: 
            for j in range(len(parm.par_name)):
                if parm.par_name[j].upper()=='GWST':
                    t1 = txtline[0:48]
                    t2 = txtline[56:len(txtline)]
                    txt = str("{:8.2f}".format(parm.cur_test_var[j]))
                    txtline = t1 + txt + t2 + '\n'
                elif parm.par_name[j].upper() == 'GWMX':
                    t1 = txtline[0:56]
                    t2 = txtline[64:len(txtline)]
                    txt = str("{:8.2f}".format(parm.cur_test_var[j]))
                    txtline = t1 + txt + t2 + '\n'
                elif parm.par_name[j].upper() == 'RFTT':
                    t1 = txtline[0:64]
                    t2 = txtline[72:len(txtline)]
                    txt = str("{:8.2f}".format(parm.cur_test_var[j]))
                    txtline = t1 + txt + t2 + '\n'
                elif parm.par_name[j].upper() == 'RFPK':
                    t1 = txtline[0:72]                    
                    txt = str("{:8.2f}".format(parm.cur_test_var[j]))
                    txtline = t1 + txt + '\n'
             
        parm.fnam2.writelines(txtline) 
        lnum += 1
            
    parm.fnam1.close()
    parm.fnam2.close()
    os.remove(fnam)
    os.rename(parm.path_TxtWork + '/' + 'temp.tmp',fnam)

    
    
 
 
