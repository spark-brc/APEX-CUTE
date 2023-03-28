import os, parm, shutil
import main_prog

def update():
    #This module updates APEX's OPC file with new set of parameters (CN and/or EFI)
     
    #Read OPC file #only one file for now
    fnam = parm.path_TxtInout + '/' + parm.fnam_opc
    fnamW = parm.path_TxtWork + '/' + parm.fnam_opc
    shutil.copyfile(fnam,fnamW)
    parm.fnam1 = open(fnamW, 'r')
    
    #Creat a temporary file
    try:
        parm.fnam2 = open(parm.path_TxtWork + '/' +'temp.tmp','w') #temp file
    except:
        #Print error message and exit
        parm.error_msg = 'Error on creating \n temp.tmp for APEX OPC file'
        msgbox.msg("Error", parm.error_msg)
        parm.iflg=1; return

    lnum = 1
    for txtline in parm.fnam1:
        txts = txtline.split()   

        if lnum > 2 and len(txts)>=8: 
           #if txts[5] == '2' and float(txts[8]) < 0:
           if float(txts[8]) < 0:
              for j in range(len(parm.par_name)):
                  if parm.par_name[j].upper()=='CN2':           
                      t1 = txtline[0:37]
                      cn2 = abs(float(txts[8]))+parm.cur_test_var[j]
                      txt = str("{:8.2f}".format(-cn2))
                      t2 = txtline[45:len(txtline)]
                      txtline = t1 + txt + t2                
                            
        parm.fnam2.writelines(txtline)         
        lnum += 1 
            
    parm.fnam1.close()
    parm.fnam2.close()
    os.remove(fnamW)
    os.rename(parm.path_TxtWork + '/' + 'temp.tmp',fnam)
 