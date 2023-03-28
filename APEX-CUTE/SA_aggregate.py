import parm, math
import numpy as np
import msgbox

def SAaggregate_data(runID):    
    p_strings = [0 for x in range(len(parm.apex_var))]              
    txt = [0 for x in range(len(parm.apex_var))]       
    txtm = [0 for x in range(len(parm.apex_var))] 

    for i in range(len(parm.apex_outlets)):              
        ndata = len(parm.pred_date[i])                

        parm.pred_datea[i] = parm.pred_date[i] 
        ival = [0 for x in range(ndata)]
        iyr = 0
        ival = parm.pred_val[i][:]
        p_strings[i] = ["%.10f" % x for x in ival]             
        meanYly = np.mean(ival)
        txtm[i] = '%.10f' % meanYly
        txt[i] = np.hstack((txtm[i],p_strings[i]))    # average annual, year1, year2, ...   
                 
    txtm = np.hstack(txtm)
    if runID>1:           
            parm.pred_fSA = np.row_stack((parm.pred_fSA,txt))              
            parm.pred_fSAm = np.row_stack((parm.pred_fSAm,txtm))# one row of output values for each model run
    else:
            parm.pred_fSA = txt  
            parm.pred_fSAm = txtm  # row 1 for model run 1
            
        # #yearly ACY
        #parm.pred_datea[i] = parm.pred_date[i]
