import os, parm, math
from numpy import genfromtxt
from shutil import copyfile 
import msgbox
import main_prog

       
def update():

    #This module updates APEX's CROP.DAT file with new set of parameters

    #copy the original crop table from back up
    fnam11 = parm.path_TxtWork + '/' + parm.fnam_crop
    fnam22 = parm.path_TxtInout + '/' + parm.fnam_crop
    os.remove(fnam11)
    copyfile(fnam22,fnam11)

    try:
        parm.fnam1 = open(fnam11, 'r+')
    except:
        #Print error message and exit
        parm.error_msg = fnam11 + ' is not found.'
        msgbox.msg("Error", parm.error_msg)
        parm.iflg=1; return 
    #Creat a temporary file
    try:
         parm.fnam2 = open(parm.path_TxtWork + '/' +'temp.tmp','w') #temp file
    except:
        #Print error message and exit
        parm.error_msg = "An error was occurredo while creating \n temp.tmp for APEX CROP file."
        msgbox.msg("Error", parm.error_msg)
        parm.iflg=1; return


    for j in range(len(parm.apex_outlets)):
        for txtline in parm.fnam1:   #Search crop table        
            for i in range(len(parm.par_name)): #Loop par_calib list
                if parm.par_filename[i]=='CROP' and parm.apex_crop[j].upper()==txtline[6:10].upper():   
                        if parm.par_name[i]=='WA':            
                            t1 = txtline[0:10]
                            t2 = txtline[18:len(txtline)]
                            var1 = float(txtline[10:18]) * parm.cur_test_var[i]
                            txt = str("{:8.3f}".format(var1))
                            txtline = t1 + txt + t2
                        elif parm.par_name[i]=='HI':            
                            t1 = txtline[0:18]
                            t2 = txtline[26:len(txtline)]
                            var1 = float(txtline[18:26]) * parm.cur_test_var[i]
                            txt = str("{:8.3f}".format(var1))
                            txtline = t1 + txt + t2
                        elif parm.par_name[i]=='TOP':            
                            t1 = txtline[0:26]
                            t2 = txtline[34:len(txtline)]
                            var1 = float(txtline[26:34]) * parm.cur_test_var[i]
                            txt = str("{:8.3f}".format(var1))
                            txtline = t1 + txt + t2
                        elif parm.par_name[i]=='TBS':            
                            t1 = txtline[0:34]
                            t2 = txtline[42:len(txtline)]
                            var1 = float(txtline[34:42]) * parm.cur_test_var[i]
                            txt = str("{:8.3f}".format(var1))
                            txtline = t1 + txt + t2
                        elif parm.par_name[i]=='DMLA':            
                            t1 = txtline[0:42]
                            t2 = txtline[50:len(txtline)]
                            var1 = float(txtline[42:50]) * parm.cur_test_var[i]
                            txt = str("{:8.3f}".format(var1))
                            txtline = t1 + txt + t2
                        elif parm.par_name[i]=='DLAI':            
                            t1 = txtline[0:50]
                            t2 = txtline[58:len(txtline)]
                            var1 = float(txtline[50:58]) * parm.cur_test_var[i]
                            txt = str("{:8.3f}".format(var1))
                            txtline = t1 + txt + t2
                        elif parm.par_name[i]=='DLAP1':            
                            t1 = txtline[0:58]
                            t2 = txtline[66:len(txtline)]
                            var1 = float(txtline[58:66])
                            var2 = math.floor(var1)
                            var3 = var1 - var2 
                            var22 = int(var2 * parm.cur_test_var[i])
                            var11 = var22 + var3 
                            txt = str("{:8.3f}".format(var11))
                            txtline = t1 + txt + t2
                        elif parm.par_name[i]=='DLAP2':            
                            t1 = txtline[0:66]
                            t2 = txtline[74:len(txtline)]
                            var1 = float(txtline[66:74])
                            var2 = math.floor(var1)
                            var3 = var1 - var2 
                            var22 = int(var2 * parm.cur_test_var[i])
                            var11 = var22 + var3 
                            txt = str("{:8.3f}".format(var11))
                            txtline = t1 + txt + t2
                        elif parm.par_name[i]=='RLAD':            
                            t1 = txtline[0:74]
                            t2 = txtline[82:len(txtline)]
                            var1 = float(txtline[74:82]) * parm.cur_test_var[i]
                            txt = str("{:8.3f}".format(var1))
                        elif parm.par_name[i]=='HMX':            
                            t1 = txtline[0:122]
                            t2 = txtline[130:len(txtline)]
                            var1 = float(txtline[122:130]) * parm.cur_test_var[i]
                            txt = str("{:8.2f}".format(var1))
                            txtline = t1 + txt + t2
                        elif parm.par_name[i]=='RDMX':            
                            t1 = txtline[0:130]
                            t2 = txtline[138:len(txtline)]
                            var1 = float(txtline[130:138]) * parm.cur_test_var[i]
                            txt = str("{:8.3f}".format(var1))
                            txtline = t1 + txt + t2
                        elif parm.par_name[i]=='BN1':            
                            t1 = txtline[0:218]
                            t2 = txtline[226:len(txtline)]
                            var1 = float(txtline[218:226]) * parm.cur_test_var[i]
                            txt = str("{:8.4f}".format(var1))
                            txtline = t1 + txt + t2
                        elif parm.par_name[i]=='BN2':            
                            t1 = txtline[0:226]
                            t2 = txtline[234:len(txtline)]
                            var1 = float(txtline[226:234]) * parm.cur_test_var[i]
                            txt = str("{:8.4f}".format(var1))
                            txtline = t1 + txt + t2
                        elif parm.par_name[i]=='BN3':            
                            t1 = txtline[0:234]
                            t2 = txtline[242:len(txtline)]
                            var1 = float(txtline[234:242]) * parm.cur_test_var[i]
                            txt = str("{:8.4f}".format(var1))
                            txtline = t1 + txt + t2
                        elif parm.par_name[i]=='BP1':            
                            t1 = txtline[0:242]
                            t2 = txtline[250:len(txtline)]
                            var1 = float(txtline[242:250]) * parm.cur_test_var[i]
                            txt = str("{:8.4f}".format(var1))
                            txtline = t1 + txt + t2
                        elif parm.par_name[i]=='BP2':            
                            t1 = txtline[0:250]
                            t2 = txtline[258:len(txtline)]
                            var1 = float(txtline[250:258]) * parm.cur_test_var[i]
                            txt = str("{:8.4f}".format(var1))
                            txtline = t1 + txt + t2
                        elif parm.par_name[i]=='BP3':            
                            t1 = txtline[0:258]
                            t2 = txtline[266:len(txtline)]
                            var1 = float(txtline[258:266]) * parm.cur_test_var[i]
                            txt = str("{:8.4f}".format(var1))
                            txtline = t1 + txt + t2
                        elif parm.par_name[i]=='FRST1':            
                            t1 = txtline[0:322]
                            t2 = txtline[330:len(txtline)]
                            var1 = float(txtline[322:330])
                            var2 = math.floor(var1)
                            var3 = var1 - var2 
                            var22 = int(var2 * parm.cur_test_var[i])
                            var11 = var22 + var3 
                            txt = str("{:8.3f}".format(var11))
                            txtline = t1 + txt + t2
                        elif parm.par_name[i]=='FRST2':            
                            t1 = txtline[0:330]
                            t2 = txtline[338:len(txtline)]
                            var1 = float(txtline[330:338])
                            var2 = math.floor(var1)
                            var3 = var1 - var2 
                            var22 = int(var2 * parm.cur_test_var[i])
                            var11 = var22 + var3 
                            txt = str("{:8.3f}".format(var11))
                            txtline = t1 + txt + t2
                        elif parm.par_name[i]=='VPTH':            
                            t1 = txtline[0:346]
                            t2 = txtline[354:len(txtline)]
                            var1 = float(txtline[346:354]) * parm.cur_test_var[i]
                            txt = str("{:8.3f}".format(var1))
                            txtline = t1 + txt + t2
                        elif parm.par_name[i]=='VPD2':            
                            t1 = txtline[0:354]
                            t2 = txtline[362:len(txtline)]
                            var1 = float(txtline[354:362]) * parm.cur_test_var[i]
                            txt = str("{:8.3f}".format(var1))
                            txtline = t1 + txt + t2
                        elif parm.par_name[i]=='RWPC1':            
                            t1 = txtline[0:362]
                            t2 = txtline[370:len(txtline)]
                            var1 = float(txtline[362:370])
                            var2 = math.floor(var1)
                            var3 = var1 - var2 
                            var22 = int(var2 * parm.cur_test_var[i])
                            var11 = var22 + var3 
                            txt = str("{:8.3f}".format(var11))
                            txtline = t1 + txt + t2
                        elif parm.par_name[i]=='RWPC2':            
                            t1 = txtline[0:370]
                            t2 = txtline[378:len(txtline)]
                            var1 = float(txtline[370:378])
                            var2 = math.floor(var1)
                            var3 = var1 - var2 
                            var22 = int(var2 * parm.cur_test_var[i])
                            var11 = var22 + var3 
                            txt = str("{:8.3f}".format(var11))
                            txtline = t1 + txt + t2
                        elif parm.par_name[i]=='BLG1':            
                            t1 = txtline[0:418]
                            t2 = txtline[426:len(txtline)]
                            var1 = float(txtline[418:426]) * parm.cur_test_var[i]
                            txt = str("{:8.3f}".format(var1))
                            txtline = t1 + txt + t2
                        elif parm.par_name[i]=='BLG2':            
                            t1 = txtline[0:426]
                            t2 = txtline[434:len(txtline)]
                            var1 = float(txtline[426:434]) * parm.cur_test_var[i]
                            txt = str("{:8.3f}".format(var1))
                            txtline = t1 + txt + t2
            
            parm.fnam2.writelines(txtline) 
            #parm.fnam1.seek(0)

    
    parm.fnam1.close()
    parm.fnam2.close()
    os.remove(fnam11)
    os.rename(parm.path_TxtWork + '/temp.tmp',fnam11)
        