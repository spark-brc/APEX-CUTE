import parm, os
from numpy import genfromtxt
import msgbox
import main_prog
import numpy as np

def read():
    parm.dds_head = []
    parm.dds_data = []
    fname = parm.path_proj + '\\dds.out'   

    if not os.path.isfile(fname):
        parm.error_msg = "Calibration output file (dds.out) does not exist in the project folder."
        msgbox.msg("Error ", parm.error_msg)
        parm.iflg=1; return

    try:
        fnam = open(fname, 'r+')
    except:
        #Print error message and exit
        parm.error_msg ="An error occured while reading dds.out."
        msgbox.msg("Error ", parm.error_msg)
        parm.iflg=1; return
    
    for txtline in fnam:
        txtline = txtline.split()
        parm.dds_head = txtline
        break 
    fnam.close()  
    
    parm.dds_data = genfromtxt(fname,delimiter='',skip_header=1) #data[row,column]

        