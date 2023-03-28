import os, parm
import numpy as np
import msgbox
import main_prog
import math

# LWE parameter range from Li eta la. 2013
def update():
    fnam = parm.path_TxtWork + '/' + parm.fnam_lwe 
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
        parm.error_msg = "An error was occurred while \n creating temp.tmp for APEXLWE file."
        msgbox.msg("Error", parm.error_msg)
        parm.iflg=1; return

    lnum = 1
    Line1 = np.zeros(10)

    for txtline in parm.fnam1:
        if lnum == 2: 
            for j in range(len(parm.par_name)):
                if parm.par_name[j].upper()=='Z0':

                    txt = str("{:<14.10f}".format(10**(parm.cur_test_var[j])))
                    t2 = txtline[14:len(txtline)]
                    txtline = txt + t2 
        elif lnum == 3:
            for j in range(len(parm.par_name)):
                if parm.par_name[j].upper()=='A':
                    txt = str("{:<14.10f}".format(parm.cur_test_var[j]))
                    t2 = txtline[14:len(txtline)]
                    txtline = txt + t2 
        elif lnum == 4:
            for j in range(len(parm.par_name)):
                if parm.par_name[j].upper()=='C':
                    txt = str("{:<14.10f}".format(parm.cur_test_var[j]))
                    t2 = txtline[14:len(txtline)]
                    txtline = txt + t2 
        elif lnum == 5:
            for j in range(len(parm.par_name)):
                if parm.par_name[j].upper()=='U-RTO':
                    txt = str("{:<14.10f}".format(parm.cur_test_var[j]))
                    t2 = txtline[14:len(txtline)]
                    txtline = txt + t2 
        elif lnum == 7:
            for j in range(len(parm.par_name)):
                if parm.par_name[j].upper()=='CY':
                    txt = str("{:<14.10f}".format(parm.cur_test_var[j]))
                    t2 = txtline[14:len(txtline)]
                    txtline = txt + t2 
        elif lnum == 8:
            for j in range(len(parm.par_name)):
                if parm.par_name[j].upper()=='PLASTP':
                    txt = str("{:<14.6f}".format(parm.cur_test_var[j]))
                    t2 = txtline[14:len(txtline)]
                    txtline = txt + t2 
        elif lnum == 9:
            for j in range(len(parm.par_name)):
                if parm.par_name[j].upper()=='KAPPA':
                    txt = str("{:<14.10f}".format(parm.cur_test_var[j]))
                    t2 = txtline[14:len(txtline)]
                    txtline = txt + t2 
             
        parm.fnam2.writelines(txtline) 
        lnum += 1
            
    parm.fnam1.close()
    parm.fnam2.close()
    os.remove(fnam)
    os.rename(parm.path_TxtWork + '/' + 'temp.tmp',fnam)

    
    
 
 
