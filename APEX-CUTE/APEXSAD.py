import parm, math, datetime
import numpy as np
import msgbox
import calendar

def select_file():
    #Read APEX output values from SAD file

    for i in range(0,len(parm.apex_outlets)):
        if parm.apex_output[i]=='SAD':
            SADread()
            break

def SADread():
    #This module reads APEX output from *.SAD file.  
    
    fnam = parm.path_TxtWork + '\\' + parm.APEXRun_name + '.SAD'
    try:
        SAD_data = np.genfromtxt(fnam,delimiter='', skip_header=10) #SAD_data[row,column]
        SAD_str = np.genfromtxt(fnam,delimiter='',dtype = 'str', skip_header=10) #SAD_data[row,column]
    except:
        #Print error message and exit
        parm.error_msg = fnam + ' is not found.'
        msgbox.msg("Error", parm.error_msg)
        parm.iflg=1; return

    #var1 = [0 for x in range(len(SAD_data))]  
    said = SAD_data[:,1]
    for i in range(0,len(parm.apex_var)):
        var1=[]
        sad_date=[]
        read_data = 1
        #check if a crop name is entered in the CUTE Settings tab
        if parm.apex_crop[i] == '' or parm.apex_var[i]<=24: 
            crop_check = 0
        else:
            crop_check = 1

        for j in range(len(said)):
            read_data = 0
            if crop_check == 1:
                if parm.apex_crop[i]==SAD_str[j,5]:
                    read_data = 1
            else:
                if j == len(said)-1:
                    read_data = 1
                else:
                    if SAD_data[j,0] != SAD_data[j+1,0]:
                        read_data = 1
                    else:
                        if SAD_data[j,4] != SAD_data[j+1,4]:
                            read_data = 1

            if read_data == 1:
                if parm.apex_outlets[i]==int(said[j]):
                    if parm.apex_var[i]==9:             #Water yield,mm
                        var1.append(SAD_data[j,23])
                    elif parm.apex_var[i]==10:             #Surface runoff,mm
                        var1.append(SAD_data[j,24])
                    elif parm.apex_var[i]==11:           #Drange flow,mm
                        var1.append(SAD_data[j,25]) 
                    elif parm.apex_var[i]==12:           #Reservior out flow, mm
                        var1.append(SAD_data[j,26]) 
                    elif parm.apex_var[i]==13:           #PET, mm
                        var1.append(SAD_data[j,27]) 
                    elif parm.apex_var[i]==14:           #ET, mm
                        var1.append(SAD_data[j,28]) 
                    elif parm.apex_var[i]==15:           #SW, mm
                        var1.append(SAD_data[j,29])
                    elif parm.apex_var[i]==16:           #SedYld, t/ha
                        var1.append(SAD_data[j,30]) 
                    elif parm.apex_var[i]==17:           #QN,kg/ha
                        var1.append(SAD_data[j,31])
                    elif parm.apex_var[i]==18:           #QP,kg/ha
                        var1.append(SAD_data[j,36])
                    elif parm.apex_var[i]==19:           #ORGN,kg/ha
                        var1.append(SAD_data[j,32])
                    elif parm.apex_var[i]==20:           #ORGP,kg/ha
                        var1.append(SAD_data[j,37])
                    elif parm.apex_var[i]==21:           #QDRN,kg/ha
                        var1.append(SAD_data[j,33])
                    elif parm.apex_var[i]==22:           #QDRP,kg/ha
                        var1.append(SAD_data[j,38])
                    elif parm.apex_var[i]==23:           #STMP, degreeC
                        var1.append(SAD_data[j,39])
                    elif parm.apex_var[i]==24:           #Biomass,t/ha
                        var1.append(SAD_data[j,10])
                    elif parm.apex_var[i]==25:           #Standing Live Biomass,t/ha
                        var1.append(SAD_data[j,11])
                    elif parm.apex_var[i]==26:           #LAI
                        var1.append(SAD_data[j,7])
                    elif parm.apex_var[i]==27:           #Crop height, m
                        var1.append(SAD_data[j,12])
                    sad_date.append(datetime.date(int(SAD_data[j,2]),int(SAD_data[j,3]),int(SAD_data[j,4])))
                
        #Aggregate output 
        if parm.obs_dt[i].upper() == 'DAILY':
            parm.pred_datea[i] = sad_date 
            parm.pred_val[i] = var1[:]

        elif parm.obs_dt[i].upper() == 'MONTHLY':     
            ndata = len(parm.pred_date[i])                
            parm.pred_datea[i] = parm.pred_date[i] 
            ival = [0 for x in range(ndata)]
            idt = 0
            sum = 0
            for j in range(len(ndata)-1):
                sum = sum + var1[j]
                if sad_date[j].month != sad_date[j+1].month:

                    if parm.apex_var[i]==15 or parm.apex_var[i]==23: #SW, STMP : average
                        curyr = sad_date[j].year
                        curmon = sad_date[j].month
                        monstr,numdays=calendar.monthrange(curyr,curmon)
                        ival[idt] = sum / numdays  
                    elif parm.apex_var[i]>=24:               #Biomass, STL, LAI, Crop height: last day value
                        ival[idt] = var1[j]
                    else:                               #all others: sum
                        ival[idt] = sum
                    idt += 1
                    sum = 0

            parm.pred_val[i] = ival[:]

        elif parm.obs_dt[i].upper() == 'YEARLY':     
            ndata = len(parm.pred_date[i])                
            parm.pred_datea[i] = parm.pred_date[i] 
            ival = [0 for x in range(ndata)]
            ival_yr = [0 for x in range(400)]
            idt = 0
            sum=0
            iday=0
            for j in range(len(var1)-1):
                sum = sum + var1[j]
                ival_yr[iday] = var1[j]
                if sad_date[j].year != sad_date[j+1].year:

                    if parm.apex_var[i]==15 or parm.apex_var[i]==23: #SW, STMP : average for the year
                        p=pd.Period(sad_date[j].year)
                        if p.is_leap_year:
                            numdays = 366
                        else:
                            numdays = 365
                        ival[idt]=sum / numdays  
                    elif parm.apex_var[i]>=24:   #Biomass, STL, LAI, Crop height: last day value of the year 
                        ival[idt] = max(ival_yr)
                    else:         #all others: sum
                        ival[idt] = sum

                                            
                    idt += 1
                    sum = 0
                    iday = 0
                iday += 1
            parm.pred_val[i] = ival[:]


        

