import parm, datetime
from numpy import genfromtxt
import msgbox

def select_file():
    #Read APEX output values from ACY file

    for i in range(0,len(parm.apex_outlets)):
        if parm.apex_output[i]=='ACY' or parm.cs100==1:
            acy_read(i)

def acy_read(i):
    #Read *.acy  

    fnam = parm.path_TxtWork + '/' + parm.APEXRun_name + '.acy'
    try:
        acy_data = genfromtxt(fnam,delimiter='',skip_header=9,dtype='unicode') #acy_data[row,column]
    except:
        #Print error message and exit
        parm.error_msg = fnam + ' is not found.'
        msgbox.msg("Error", parm.error_msg)
        parm.iflg=1; return

    inum = acy_data.shape[0]
    iyr = [0 for x in range(inum)] 
    isub = [0 for x in range(inum)] 
    icp = [0 for x in range(inum)] 
    ival = [0 for x in range(inum)] 

    #read Sub and Crop names from ACY
    isub=acy_data[:,1]
    icp=acy_data[:,4]

    #Read values
    icount = 0
    for j in range(inum):
        if int(isub[j])==parm.apex_outlets[i] and icp[j].upper()==parm.apex_crop[i].upper():
            if parm.apex_var[i]==28: #Grain yield,ton/ha/yr:
                iyr[icount] = datetime.date(int(acy_data[j,2]),1,1)
                ival[icount] = float(acy_data[j,5])
                icount += 1
       
    parm.pred_date[i] = iyr[0:icount]
    parm.pred_datea[i] = parm.pred_date[i]  #date            
    parm.pred_val[i] = ival[0:icount]

    #Read annual average yield for constraints
    if parm.cs100==1:
        for i in range(len(parm.cs_name)):
            icount = 0
            yld = 0
            if parm.cs_type[i]==100:
                for j in range(inum):
                    if icp[j].upper()==parm.cs_name[i].upper():
                        yld = yld + float(acy_data[j,5])
                        icount += 1
        
                parm.crop_yld.append(yld / icount)


