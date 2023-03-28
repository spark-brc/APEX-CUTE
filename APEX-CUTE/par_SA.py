import parm
import math

def read():
    fnam = parm.path_proj + '\\Params.txt' #params file for SA sampling
    fcon = open(fnam,'w')
    for i in range(len(parm.par_name)):
        t1 = parm.par_name[i] 
        if t1 != 'mu' and t1 != 'sigma':
            parm.NumPar_fSA += 1                            
            t2 = parm.par_bl[i]
            t3 = parm.par_bu[i]

            if t1.upper() == 'Z0':
                if t2>0: 
                    t2 = math.log10(t2)
                else:
                    t2 = -7
                if t3>0:
                    t3 = math.log10(t3)
                else:
                    t3 = -1

            txt = t1 + ' ' + str(t2) + ' ' + str(t3) +'\n'
            fcon.writelines(txt)                                   

    fcon.close()

