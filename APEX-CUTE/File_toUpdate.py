import parm

#Identify APEX input files to update
def update():
    j=0
    parm.iparm=0
    parm.icont=0
    parm.isite=0
    parm.isuba=0
    parm.idly=0
    parm.iopsc=0
    parm.isoil=0
    parm.icrop=0
    parm.ilwe=0
    for j in range(len(parm.par_name)):
        if parm.par_filename[j].upper()=='PARM':
            parm.iparm = parm.iparm + 1
        elif parm.par_filename[j].upper()=='APEXCONT':
            parm.icont = parm.icont + 1
        elif parm.par_filename[j].upper()=='SIT':
            parm.isite = parm.isite + 1
        elif parm.par_filename[j].upper()=='SUB':
            parm.isuba = parm.isuba + 1
        elif parm.par_filename[j].upper()=='OPC' or parm.par_filename[j].upper()=='OPS':
            parm.iopsc = parm.iopsc + 1
        elif parm.par_filename[j].upper()=='SOL':
            parm.isoil = parm.isoil + 1  # remember there may be multiple .sol files
        elif parm.par_filename[j].upper()=='CROP':
            parm.icrop = parm.icrop + 1
        elif parm.par_filename[j].upper()=='LWE':
            parm.ilwe = parm.ilwe + 1  