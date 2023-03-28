import parm, os, subprocess, math, pcp
import numpy as np


def run():
    import APEXPARM, APEXSITE, APEXSUBA, APEXACY, data_pairing, statistics
    import APEXOPSC, APEXSOIL, APEXRCH, APEXSAD, APEXCROP, APEXCONT, APEXDWS, apexLWE, LWEOUT
         
    #Update APEX files 
    if parm.iparm > 0:
        APEXPARM.update()
        if parm.iflg==1: return                 
    #if parm.icont > 0:
    APEXCONT.update() # Update parameters
    if parm.iflg==1: return                 
    if parm.isuba > 0:
        APEXSUBA.update()
        if parm.iflg==1: return                 
    if parm.isite > 0:
        APEXSITE.update()
        if parm.iflg==1: return                 
    if parm.iopsc > 0:
        APEXOPSC.update()
        if parm.iflg==1: return                 
    #if parm.isoil > 0:
    #    APEXSOIL.update()
    if parm.idly > 0:
        pcp.update(parm.txt_iyr)
        if parm.iflg==1: return                 
    if parm.icrop > 0:
        APEXCROP.update() 
        if parm.iflg==1: return                 
    if parm.ilwe > 0:
        apexLWE.update() 
        if parm.iflg==1: return                 


    #Run APEX
    os.chdir(parm.path_TxtWork)
 
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    retcode = subprocess.Popen('apex1501.exe',startupinfo=startupinfo,creationflags=0x08000000)
    retcode.wait()

    #Collect APEX output     
    APEXRCH.select_file()
    if parm.iflg==1: return                 
    APEXSAD.select_file()
    if parm.iflg==1: return                 
    APEXDWS.select_file()
    if parm.iflg==1: return                 
    APEXACY.select_file()
    if parm.iflg==1: return                 
    LWEOUT.select_file()
    if parm.iflg==1: return                 
      
    #Arrange paired dataset for pred-obs comparison
    data_pairing.dataPair()    
    if parm.iflg==1: return                 

    #Calculate OF
    parm.cur_test_OF = 0
    sum_weight = 0 
    for i in range(len(parm.apex_outlets)):
        sum_weight = sum_weight + parm.of_weight[i]

    if parm.obs_yld_type=='9999':
        if len(parm.pairpCal)<6:
            ofnew = (np.mean(parm.pairoCal) - np.mean(parm.pairpCal)) / np.mean(parm.pairoCal) * 100 #percent bias
        else:
            statistics.performance_indicators(parm.pairpCal,parm.pairoCal)
            ofnew = 0
            if parm.dds_stat.upper()=="F(NSE-PBIAS)":
                ofnew = math.sqrt((1-parm.nse)**2+(math.fabs(parm.re)/100+0.5)**2)
            elif parm.dds_stat.upper()=="R2":
                ofnew = 1 - parm.r2
            elif parm.dds_stat.upper()=="RMSE":
                ofnew = parm.rmse
            elif parm.dds_stat.upper()=="ABSOLUTE ERROR":
                ofnew = parm.bias
            elif parm.dds_stat.upper()=="PBIAS":
                ofnew = math.fabs(parm.re)
            elif parm.dds_stat.upper()=="NSE":
                ofnew = 1 - parm.nse
        parm.cur_test_OF = ofnew
    else:
        for i in range(len(parm.apex_outlets)):
            parm.of_weight[i] = parm.of_weight[i] / sum_weight
            statistics.performance_indicators(parm.pairpCal[i],parm.pairoCal[i])
            ofnew = 0
            if parm.dds_stat.upper()=="F(NSE-PBIAS)":
                ofnew = math.sqrt((1-parm.nse)**2+(math.fabs(parm.re)/100+0.5)**2)
            elif parm.dds_stat.upper()=="R2":
                ofnew = 1 - parm.r2
            elif parm.dds_stat.upper()=="RMSE":
                ofnew = parm.rmse
            elif parm.dds_stat.upper()=="ABSOLUTE ERROR":
                ofnew = parm.bias
            elif parm.dds_stat.upper()=="PBIAS":
                ofnew = math.fabs(parm.re)
            elif parm.dds_stat.upper()=="NSE":
                ofnew = 1 - parm.nse

            parm.cur_test_OF = parm.cur_test_OF + ofnew * parm.of_weight[i]