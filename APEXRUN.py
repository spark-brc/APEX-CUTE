import parm
import msgbox
import main_prog


def read():
        
    fnam = parm.path_TxtWork + '\APEXRUN.DAT'
    try:
        parm.fnam1 = open(fnam, 'r')
    except:
        #Print error message and exit
        parm.error_msg = fnam + ' is not found.'
        msgbox.msg("Error", parm.error_msg)
        parm.iflg=1; return
 
    lnum = 1
    for txtline in parm.fnam1:
       txtline = txtline.split()
       if lnum == 1: 
            parm.APEXRun_name = txtline[0]  # Name for APEX run
            parm.ISIT = txtline[1]  #Site file number
            parm.ISUB = txtline[4]  #Subarea file number  
            lnum += 1
       else:
             break 
           
    parm.fnam1.close()  
