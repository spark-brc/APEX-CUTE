import os, parm
import msgbox
import main_prog


def update():
    #This module updates *.SUB file with a new PEC (erosion control practice factor) value
 
    #Read *.sub file
    fnam = parm.path_TxtWork + '/' + parm.fnam_sub
    try:
        parm.fnam1 = open(fnam, 'r+')
    except:
        #Print error message and exit
        parm.error_msg = fnam + ' is not found.'
        msgbox.msg("Error", parm.error_msg)
        parm.iflg=1; return
   
    #Creat a temporary file
    try:
         parm.fnam2 = open(parm.path_TxtWork + 'temp.tmp','w') #temp file
    except:
        #Print error message and exit
        parm.error_msg = "An error was occurred while \n creating temp.tmp for  APEX SUB file."
        msgbox.msg("Error", parm.error_msg)
        parm.iflg=1; return


    lnum = 1
    for txtline in parm.fnam1:
        if lnum == 10:
            for j in range(len(parm.par_name)):
                if parm.par_name[j].upper() == 'PEC': 
                    t1 = str("{:10.2f}".format(parm.cur_test_var[j]))                    
                    t2 = txtline[10:len(txtline)]
                    txtline = t1 + t2 
        
        parm.fnam2.writelines(txtline) 
        lnum += 1
            
    parm.fnam1.close()
    parm.fnam2.close()
    os.remove(fnam)
    os.rename(parm.path_TxtWork + 'temp.tmp',fnam)
    