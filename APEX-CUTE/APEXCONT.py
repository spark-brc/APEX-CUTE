import os, parm, datetime
import numpy as np
import msgbox
import main_prog

def read():
    idate=[]
    #read APEXCONT.DAT to initiate the program    
    fnam = parm.path_TxtWork + '\APEXCONT.DAT'
    try:
        parm.apexcont = open(fnam, 'r')
    except:
        #Print error message and exit
        parm.error_msg = fnam + ' is not found.'
        msgbox.msg("Error", parm.error_msg)
        parm.iflg=1; return 


    lnum = 1
    for txtline in parm.apexcont:
        if lnum == 1: 
            dum = ' '.join(txtline.split())
            data = dum.split(' ')
            parm.txt_nbyr = int(float(data[0]))   #NBYR: Number of years simulated.
            parm.txt_iyr = int(float(data[1]))    #IYR: Beginning year of simulation.
            parm.txt_imo = int(float(data[2]))   #IMO: Beginning month of simulation.
            parm.txt_ida = int(float(data[3]))  #IDA: Beginning julian day of simulation.
            parm.txt_ipd = int(float(data[4]))  #IPD: Print code for type of output in .rch.
            parm.txt_iet = int(float(data[9]))  #IET: PET Method
            parm.txt_nvcn = int(float(data[14])) #
            parm.txt_infl = int(float(data[15])) #INFL=1: Green & Ampt estimate of Q.
        
        elif lnum == 6:
             parm.txt_drv = int(float(txtline[0:8])) 

        lnum += 1
            
    parm.apexcont.close()
    
    a = datetime.date(parm.txt_iyr, parm.txt_imo, parm.txt_ida) 
    b = datetime.date((parm.txt_iyr+parm.txt_nbyr-1), 12, 31)   
    for i in range(0,len(parm.apex_outlets)):
        if parm.obs_dt[i].upper() == 'DAILY':
            day_count = (b-a).days+1
            idate = [0 for x in range(day_count)] 
            idate[0] = a
            for ii in range(1,day_count):
                idate[ii] = idate[ii-1] + datetime.timedelta(days=1)

        elif parm.obs_dt[i].upper() == 'MONTHLY':     
            day_count = (b.year - a.year) * 12 + b.month - a.month + 1
            idate = [0 for x in range(day_count)] 
            idate[0] = a
            for ii in range(1,day_count):
                month = idate[0].month + ii - 1
                year = idate[0].year + month // 12
                month = month % 12 + 1
                idate[ii] = datetime.date(year, month, 1)

        else: # YEARLY 
            day_count = (b.year - a.year)  + 1
            idate = [0 for x in range(day_count)] 
            idate[0] = a
            for ii in range(1,day_count):
                idate[ii] = addYears(idate[ii-1],1)

        parm.pred_date[i] = idate
        parm.start_pred[i] = idate[0]
        parm.end_pred[i] = idate[len(idate)-1] 
    
        #Check periods of simulation, calibration, and validation 
        if parm.SA_orCal == 0: # for calibration
            if parm.end_cal < parm.start_pred[i] or parm.start_cal > parm.end_pred[i]:
                parm.error_msg = 'Calibration period is not within the simulation period!!'    
                msgbox.msg("Warning",parm.error_msg) 
                parm.iflg=1; return 
       
            if parm.flg_validation!=0:
                if parm.end_val < parm.start_pred[i] or parm.start_val > parm.end_pred[i]:
                    parm.error_msg = 'Validation period is not within the simulation period!!'    
                    msgbox.msg("Warning",parm.error_msg) 
                    parm.iflg=1; return 

def addYears(d, years):
    try:
#Return same day of the current year        
        return d.replace(year = d.year + years)
    except ValueError:
#If not same day, it will return other, i.e.  February 29 to March 1 etc.        
        return d + (date(d.year + years, 1, 1) - date(d.year, 1, 1))
            
def update(): 
    #update APEXCONT.DAT       
    fnam = parm.path_TxtWork + '\APEXCONT.DAT'
    try:
        parm.fnam1 = open(fnam, 'r')
    except:
        parm.error_msg = fnam + ' is not found.'
        msgbox.msg("Error", parm.error_msg)
        parm.iflg=1; return 

    #Creat a temporary file
    try:
        ftmp =  parm.path_TxtWork + '/' +'temp.tmp'
        parm.fnam2 = open(ftmp,'w') #temp file
    except:
        #Print error message and exit
        parm.error_msg = "An error was occurredo when creating temp.tmp for APEXCONT file."
        msgbox.msg("Error",parm.error_msg )
        parm.iflg=1; return 
          
    #update apexcont.dat
 
    # %%%%%%%%%%%%%%%%%%%%  Amir Sharifi additions , 06-09-2015 -----  Lines 3,4,5 and 6 of APEXCONT will be accessible and modifiable %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    plist =    ('RFN',	'CO2',	'CQN',	'PSTX',	'YWI',	'BTA',	'EXPK',	'QG',	'QCF',	'CHSO',
                'BWD',	'FCW',	'FPSC',	'GWSO',	'RFTO',	'RFPO',	'SATO',	'FL',	'FW',	'ANG',
                'UXP',	'DIAM',	'ACW',	'GZL0',	'RTN0',	'BXCT',	'BYCT',	'DTHY',	'QTH',	'STND',
                'DRV',	'PCO0',	'RCC0',	'CSLT',	'CPV0',	'CPH0',	'BUS(1)', 'BUS(2)' , 'BUS(3)', 'BUS(4)')
    nparm = len(plist)
    prmt = ['' for x in range(0,nparm)]

    linestr = parm.fnam1.readlines()
 
    # Reset printing time interval (IPD) to daily time step (IPD=6)
    ts = 6
    #ipd = [0 for x in range(len(parm.apex_var))]
    #for i in range(len(parm.apex_var)):
    #    if parm.apex_var[0]<=8: #.rch output to evaluate
    #        if parm.obs_dt[i].upper() == 'DAILY':
    #            ipd[i] = 6
    #        elif parm.obs_dt[i].upper() == 'MONTHLY':     
    #            ipd[i] = 3
    #        else: 
    #            ipd[i] = 1

    #    if ts < ipd[i]: ts = ipd[i]

    #update IPD if IPD is sparser than txt_ipd 
    txtline = linestr[0]
    dum = ' '.join(txtline.split())   #updated for APEX1501 Nov. 2015
    data = dum.split(' ')
    txt = '  ' + data[0] + ' ' + data[1] + '  ' + data[2] + '  ' + data[3] 
    txt = txt + str("{:3}".format(ts)) + ' ' + data[5]
    for i in range(6,len(data)):
        txt = txt + '   ' + data[i]               
        
    linestr[0]=txt + '\n'
 
    #read parm values from apexcont.dat
    for i in range(0,4):
        txtline = linestr[i+2]
        dum = ' '.join(txtline.split())   #updated for APEX1501 May 2016 
        data = dum.split(' ')
        for j in range (0,10):
            k = i * 10 + j 
            prmt[k] = data[j]

            if k == nparm - 1 or j==len(data)-1: break

    #update values for selected parms
    for k in range(0,nparm):
        for j in range(len(parm.par_name)):
            if parm.par_name[j].upper() == plist[k]:
                prmt[k] = str(parm.cur_test_var[j])
                break
   
    #update each line 
    for i in range(0,4):
        txt = ''
        for j in range(0,10):
            k = i * 10 + j
            if k == nparm or prmt[k]=='': break
            txt = txt + "{0:>8.2f}".format(float(prmt[k]))   

        txt = txt + '\n'
        linestr[i+2] = txt

    #reprint PARM.DAT       
    parm.fnam2.writelines(linestr[0:len(linestr)])

    parm.fnam1.close()
    parm.fnam2.close()
    os.remove(fnam)
    os.rename(ftmp,fnam)