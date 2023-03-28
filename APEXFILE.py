import parm
import msgbox
import main_prog


def read():
    #Read APEXFILE.DAT and get file names for *.sit, *.sub, *.ops/c, and *.sol
        
    fname = parm.path_TxtWork + '\APEXFILE.DAT'
    try:
        fcon = open(fname, 'r')
    except:
        #Print error message and exit
        parm.error_msg = fnam + ' is not found.'
        msgbox.msg("Error", parm.error_msg)
        parm.iflg=1; return

    lnum = 0
    for txtline in fcon:
        txtline = txtline.split()
        lnum += 1

        if lnum == 1: #APEX Site table
            FSITE = txtline[1]
                        
        elif lnum == 2: #APEX Subarea table
            FSUBA = txtline[1]

        elif lnum == 5: #APEX crop file
            parm.fnam_crop = txtline[1]

        elif lnum == 9: #APEX SOIL table
            FSOIL = txtline[1]
                                    
        elif lnum == 10: #APEX OPSC table
            FOPSC = txtline[1]

        elif lnum == 12: #APEX PARM file
            parm.fnam_parm = txtline[1]

        elif lnum == 14: #APEX Print file
            parm.fnam_print = txtline[1]

        elif lnum == 19: #APEX LWE file
            parm.fnam_lwe = txtline[1]

    fcon.close()

    if parm.isite > 0:
        parm.fnam_sit = readFileNam(FSITE,parm.ISIT)    

    rsub = 0
    if parm.isuba > 0:  
        parm.fnam_sub = readFileNam(FSUBA,parm.ISUB) 
        rsub = 1
        
    # Listing No. of .opc and/or .sol file/s provided in .sub file for each individual subarea.
    # Now coded only for one subarea.
    if parm.iopsc > 0 or parm.isoil > 0: 
        if rsub < 1:
            parm.fnam_sub = readFileNam(FSUBA,parm.ISUB)

        #Read .sol and/or .opc/s file listing ID/s from *.sub file.        
        parm.fnam1 = open(parm.path_TxtWork + '/' + parm.fnam_sub, 'r')

        lnum = 0
        for txtline in parm.fnam1:
            txtline = txtline.split()
            lnum += 1 
                       
            if lnum == 2:                 
                if parm.isoil > 0: 
                    parm.ISOL = txtline[0] #List ID for the *.sol                
                    parm.fnam_sol = readFileNam(FSOIL,parm.ISOL)

                if parm.iopsc > 0: 
                    parm.IOPS = txtline[1] #List ID for the *.opc
                    parm.fnam_opc = readFileNam(FOPSC,parm.IOPS)
                break

        parm.fnam1.close()

def readFileNam(fileTable,id):
    #Read the name of *.sub from FSUBA
    fnam = parm.path_TxtWork + '/' + fileTable
    try:
         parm.fnam1 = open(fnam, 'r')
    except:
        #Print error message and exit
        parm.error_msg = fnam + ' is not found.'
        msgbox.msg("Error", parm.error_msg)
        parm.iflg=1; return
    
    for txtline in parm.fnam1:
         txtline = txtline.split()
         if txtline[0]==id: 
             #parm.fnam_sub = txtline[1] #Name of file
             fileNam = txtline[1] #Name of file
             break

    parm.fnam1.close()
    return fileNam   