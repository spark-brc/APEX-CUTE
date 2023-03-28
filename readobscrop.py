import parm, datetime, os, os.path
from numpy import genfromtxt
import msgbox

def read(i):
    # i is the subarea ID in the Outlets tab
    #Reads yearly measured crop yield data 
    filelist = []   
    filename = parm.path_obs + '\obs_crop.csv'
    file_exists = os.path.exists(filename)
    if file_exists == True:
        for file in os.listdir(parm.path_obs):
            if file.endswith(".csv"):
                filelist.append(file)
    else:
        parm.error_msg = filename + ' is not found'
        msgbox.msg("Error:", parm.error_msg)
        parm.iflg=1
        return

    fcon = open(filename, 'r')
    lnum = 0
    yld_type = 0
    for txtline in fcon:
        txtline = txtline.split(',')
        lnum += 1
        if lnum>1:
            parm.obs_yld_type = txtline[0]
            break
    fcon.close()

    if parm.obs_yld_type=='9999': #Average annual yield
        read_avg(i)
    else:
        read_timeseriese(i)

def read_avg(i):
    #read observed crop yield data given as average annual value
    isub = parm.apex_outlets[i]
    filename = parm.path_obs + '\obs_crop.csv'
    obsdata = genfromtxt(filename,  delimiter=',', skip_header=1)
    inum = obsdata.shape[0]
    icp = [0 for x in range(inum)] 
    sub = [0 for x in range(inum)] 
    iyld = [0 for x in range(inum)] 
    ibiom = [0 for x in range(inum)] 

    #read Crop names
    fcon = open(filename, 'r')
    lnum = 0
    for txtline in fcon:
        txtline = txtline.split(',')
        lnum += 1
        if lnum>1:
            sub[lnum-2] = int(float(txtline[1]))
            icp[lnum-2] = txtline[2]
            iyld[lnum-2] = float(txtline[3])
            ibiom[lnum-2] = float(txtline[4])

    fcon.close()

    for j in range(inum):
        if sub[j]==isub and icp[j].upper()==parm.apex_crop[i].upper():
            if parm.apex_var[i]==28:  #Grain yield,ton/ha/yr:
                parm.obs_val[i] = iyld[j]
            #elif parm.apex_var[i]==29: #Biomass yield,ton/ha/yr
            #    parm.obs_val[i] = ibiom[j] 
        
    parm.obs_date[i] = 9999


def read_timeseriese(i):
    isub = parm.apex_outlets[i]
    #read time series crop yield data
    filename = parm.path_obs + '\obs_crop.csv'
    obsdata = genfromtxt(filename,  delimiter=',', skip_header=1,dtype='unicode')
    inum = int(obsdata.shape[0])
    iyr = [0 for x in range(inum)] 
    icp = [0 for x in range(inum)] 
    ival = [0 for x in range(inum)] 

    #read Crop names
    fcon = open(filename, 'r')
    lnum = 0
    for txtline in fcon:
        txtline = txtline.split(',')
        if lnum>=1:
            icp[lnum-1] = txtline[2]
        lnum += 1
    fcon.close()

    #Read values
    icount = 0
    #annual crop yields from obs_crop.csv
    for j in range(inum):
        if int(obsdata[j,1])==isub and icp[j].upper()==parm.apex_crop[i].upper():
            iyr[icount] = datetime.date(int(obsdata[j,0]),1,1)
            ival[icount] = float(obsdata[j,3])#Grain yield,ton/ha/yr
            icount += 1

                        
    parm.obs_date[i] = iyr[0:icount]
    parm.obs_val[i] = ival[0:icount]


