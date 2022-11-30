import parm, os, subprocess

def run(runID):
    import APEXPARM, APEXSITE, APEXSUBA, SA_aggregate, data_pairing
    import APEXOPSC, APEXSOIL, APEXCROP, APEXCONT, APEXRCH, APEXACY, APEXSAD, apexLWE, LWEOUT
#    import SA_TEST
#    import APEXACY
         
    #Update APEX files 
    if parm.iparm>0:
        APEXPARM.update()
        if parm.iflg==1: return                 
  #  if parm.icont>0:
    APEXCONT.update()
    if parm.iflg==1: return                 
    if parm.isuba>0:
        APEXSUBA.update()
        if parm.iflg==1: return                 
    if parm.isite>0:
        APEXSITE.update()
        if parm.iflg==1: return                 
    if parm.iopsc>0:
        APEXOPSC.update()
        if parm.iflg==1: return                 
    if parm.isoil > 0:
        APEXSOIL.update() 
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
    APEXACY.select_file()
    if parm.iflg==1: return 
    LWEOUT.select_file()
    if parm.iflg==1: return                 
   
    #get the simulation period
    if parm.iflg==1: return 

    SA_aggregate.SAaggregate_data(runID)
    if parm.iflg==1: return                 
#    APEXACY.select_file()    
#    acyread()

