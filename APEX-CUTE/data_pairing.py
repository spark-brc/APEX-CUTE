import sys, parm, os, subprocess, datetime, math 
from numpy import genfromtxt
import msgbox

def run():
         
    #Allocate some variables          
    parm.pairdCal = [0 for x in range(len(parm.apex_outlets))]
    parm.pairpCal = [0 for x in range(len(parm.apex_outlets))]
    parm.pairoCal = [0 for x in range(len(parm.apex_outlets))]
    parm.pairdVal = [0 for x in range(len(parm.apex_outlets))]       
    parm.pairpVal = [0 for x in range(len(parm.apex_outlets))]
    parm.pairoVal = [0 for x in range(len(parm.apex_outlets))]

    #Prepare for date pairing
    for i in range(len(parm.apex_outlets)):
        #Aggregate daily output to monthly/yearly time interval
        if parm.obs_dt[i].upper()=='DAILY': #daily
            parm.pred_datea[i] = parm.pred_date[i]  #date            
        elif parm.obs_dt[i].upper()=='MONTHLY': #monthly
            nmon = (parm.end_pred[i].year - parm.start_pred[i].year) * 12 + parm.end_pred[i].month - parm.start_pred[i].month + 1            
            imon = 0
            idate = [0 for x in range(nmon)]
            idate[0] = datetime.date(parm.start_pred[i].year,parm.start_pred[i].month,1) 
            for j in range(len(parm.pred_date[i])-1):
                if parm.pred_date[i][j].month!=parm.pred_date[i][j+1].month:
                    imon += 1             
                    idate[imon] = datetime.date(parm.pred_date[i][j+1].year,parm.pred_date[i][j+1].month,1)  
                    
            parm.pred_datea[i] = idate[:]

        elif parm.obs_dt[i].upper()=='YEARLY' and parm.apex_output[i]!='ACY': #yearly            
            nyr = parm.txt_nbyr
            idate = [0 for x in range(nyr)]
            iyr = 0
            idate[0] = datetime.date(parm.start_pred[i].year,1,1) 
            for j in range(len(parm.pred_date[i])-1):
                if parm.pred_date[i][j].year!= parm.pred_date[i][j+1].year:
                    iyr += 1
                    idate[iyr] = datetime.date(parm.pred_date[i][j+1].year,1,1)  

            parm.pred_datea[i] = idate[:]

        else: #yearly ACY
            parm.pred_datea[i] = parm.pred_date[i]  #date            

        #Compare date periods between pred/obs/calibration
        if parm.SA_orCal != 1: 
            parm.pairdCal[i] = pairdateCalVal(i, parm.start_cal, parm.end_cal)
            if len(parm.pairdCal[i]) == 0:
                parm.error_msg = 'There is no observed data \n for calibration period.'
                msgbox.msg("Error",parm.error_msg)
                parm.iflg=1; return

            if parm.flg_validation==1:
                parm.pairdVal[i] = pairdateCalVal(i, parm.start_val, parm.end_val)   
                if len(parm.pairdVal[i]) == 0:
                    parm.error_msg = 'There is no observed data \n for validation period.'
                    msgbox.msg("Error",parm.error_msg)
                    parm.iflg=1; return

def pairdateCalVal(i, start_p, end_p):
    days = [start_p, parm.start_obs[i],parm.start_pred[i]]
    first_day = max(days)
    days = [end_p,parm.end_obs[i],parm.end_pred[i]]
    end_day = min(days)

    #Duration of the paired data
    if parm.obs_dt[i].upper()=='DAILY':   #daily
        data_due = (end_day - first_day).days+1            
    elif parm.obs_dt[i].upper()=='MONTHLY': #monthly
        data_due = (end_day.year - first_day.year) * 12 + end_day.month - first_day.month + 1            
    else:                             #yearly
        data_due = end_day.year - first_day.year + 1                

    #Pair data for calibration and/or validation periods with corresponding obs.
    if not parm.obs_date[i]==9999:
        pairdate = [0 for x in range(data_due)]                
        icount = 0
        for ip in range(len(parm.pred_datea[i])):
            if parm.pred_datea[i][ip]>=first_day and parm.pred_datea[i][ip]<=end_day:
                for io in range(len(parm.obs_date[i])):
                    if parm.obs_date[i][io]>=first_day and parm.obs_date[i][io]<=end_day:
                       if parm.pred_datea[i][ip]==parm.obs_date[i][io]:
                           if not parm.obs_val[i][io]<=-99 or parm.obs_val[i][io]>=9999: #skip no obs data date
                               pairdate[icount] = parm.pred_datea[i][ip]
                               icount += 1
                               break
    else:
        pairdate = []
        pairdate.append(parm.obs_date[i])
        icount = 1

    return pairdate[0:icount] 

def dataPair():
    #aggregate predicted output to monthly or yearly time interval
    nmon,nyr=0,0
    for i in range(len(parm.apex_outlets)):
        if parm.obs_dt[i].upper()=='DAILY': #daily
            parm.pred_vala[i] = parm.pred_val[i]
        elif parm.obs_dt[i].upper()=='MONTHLY': #monthly
            nmon = (parm.end_pred[i].year - parm.start_pred[i].year) * 12 + parm.end_pred[i].month - parm.start_pred[i].month + 1
            ival = [0 for x in range(nmon)]
            imon = 0            
            for j in range(len(parm.pred_date[i])-1):
                if parm.pred_date[i][j].month!=parm.pred_date[i][j+1].month:
                    imon += 1
                ival[imon] = ival[imon] + parm.pred_val[i][j]

            parm.pred_vala[i] = ival[:]
            
        elif parm.obs_dt[i].upper()=='YEARLY' and parm.apex_var[i]!=28: #yearly 
            nyr = parm.txt_nbyr
            ival = [0 for x in range(nyr)]
            iyr = 0
            for j in range(len(parm.pred_date[i])-1):
                if parm.pred_date[i][j].year!=parm.pred_date[i][j+1].year:
                    iyr += 1
                ival[iyr] = ival[iyr] + parm.pred_val[i][j]

            parm.pred_vala[i] = ival[:]

        else: #yearly ACY or avg annual yld
            if parm.obs_yld_type=='9999':
                parm.pred_vala[i] = sum(parm.pred_val[i][:]) / len(parm.pred_val[i])
            else:
                parm.pred_vala[i] = parm.pred_val[i][:]

        #Compare data periods between pred/obs/calibration
        cali = 1
        pairdataCalVal(cali, i, parm.start_cal, parm.end_cal, parm.pairdCal[i])
        if parm.flg_validation==1:
            cali = 0
            pairdataCalVal(cali, i, parm.start_val, parm.end_val, parm.pairdVal[i]) 

def pairdataCalVal(cali, i, start_p, end_p, pd):

        if parm.obs_date[i]==9999:
            if cali: 
                parm.pairpCal[i] = parm.pred_vala[i]
                parm.pairoCal[i] = parm.obs_val[i]
            else:
                parm.pairpVal[i] = parm.pred_vala[i]
                parm.pairoVal[i] = parm.obs_val[i]            
        else:
            #Compare data periods between pred/obs/calibration
            days = [start_p,parm.start_obs[i],parm.start_pred[i]]
            first_day = max(days)             
            days = [end_p,parm.end_obs[i],parm.end_pred[i]]
            end_day = min(days)        

            pair1 = [0 for x in range(len(pd))]  
            pair2 = [0 for x in range(len(pd))]

            kk=0
            for ip in range(len(parm.pred_datea[i])):
                if parm.pred_datea[i][ip]>=first_day and parm.pred_datea[i][ip]<=end_day:
                    for io in range(len(parm.obs_date[i])):
                        if parm.obs_date[i][io]>=first_day and parm.obs_date[i][io]<=end_day:
                            if parm.pred_datea[i][ip]==parm.obs_date[i][io]:
                                if not parm.obs_val[i][io]<-99 or parm.obs_val[i][io]>=9999: #No data
                                    pair1[kk] = parm.pred_vala[i][ip]
                                    pair2[kk] = parm.obs_val[i][io]                                   
                                    kk+=1
                                    break
                        if parm.obs_date[i][io] > end_day:
                            break
                if parm.pred_datea[i][ip] > end_day:
                    break

            #paired data for predicted and observed values
            if cali: 
                parm.pairpCal[i] = pair1[:]
                parm.pairoCal[i] = pair2[:]
            else:
                parm.pairpVal[i] = pair1[:]
                parm.pairoVal[i] = pair2[:]
