import os, parm
import msgbox
import main_prog


def update():
    #This module updates *.sit file with a new APM (peak runoff rate - rainfall energy adjustment factor) value    
    
    #Read *.sit file
    fnam = parm.path_TxtWork + '/' + parm.fnam_sit
    try:
        parm.fnam1 = open(fnam, 'r+')
    except:
        #Print error message and exit
        parm.error_msg = fnam + ' is not found.'
        msgbox.msg("Error", parm.error_msg)
        parm.iflg=1; return

    parm.fnam2 = open(parm.path_TxtWork + 'temp.tmp','w') #temp file
    
    lnum = 1
    for txtline in parm.fnam1:
        if lnum == 4: 
            for j in range(len(parm.par_name)):
                if parm.par_name[j].upper() == 'APM':
                    t1 = txtline[0:24]
                    t2 = txtline[32:len(txtline)]
                    txt = str("{:8.2f}".format(parm.cur_test_var[j]))
                    txtline = t1 + txt + t2 
        
        parm.fnam2.writelines(txtline) 
        lnum += 1
            
    parm.fnam1.close()
    parm.fnam2.close()
    os.remove(fnam)
    os.rename(parm.path_TxtWork + 'temp.tmp',fnam)

    
    
 
 
