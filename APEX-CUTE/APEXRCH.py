import parm
from numpy import genfromtxt
import msgbox
import main_prog


def select_file():
    #Read APEX output values from RCH file

    for i in range(0,len(parm.apex_outlets)):
        if parm.apex_output[i]=='RCH':
            rch_read()
            break
        
def rch_read():

    fnam = parm.path_TxtWork + '/' + parm.APEXRun_name + '.RCH'   
    try:
        parm.fnam1 = open(fnam, 'r') #rch_data[row,column]
    except:
        #Print error message and exit
        parm.error_msg = fnam + ' is not found.'
        msgbox.msg("Error", parm.error_msg)
        parm.iflg=1; return

    linestr= parm.fnam1.readlines()

    for i in range(0,len(parm.apex_var)):
        var1 = [0 for x in range(len(parm.pred_datea[i]))]        
        icount = 0

        for j in range(9,len(linestr)):
            txt=linestr[j]
            dum = ' '.join(txt.split())
            rch_data=dum.split(' ')

            if parm.apex_outlets[i]==int(float(rch_data[1])):
                icount += 1  
                wsa = float(rch_data[4]) #drainage area,ha
                if parm.apex_var[i]==0:             #flow,m3/s
                    var1[icount-1] = float(rch_data[8]) 
                elif parm.apex_var[i]==1:           #sediment,tons
                    var1[icount-1] = float(rch_data[12]) 
                elif parm.apex_var[i]==2:           #TN,kg
                    var1[icount-1] = float(rch_data[15])+float(rch_data[19])+float(rch_data[21])+float(rch_data[23]) 
                    var1[icount-1] = float(var1[icount-1])
                elif parm.apex_var[i]==3:           #TP,kg
                    var1[icount-1] = float(rch_data[17])+float(rch_data[25]) 
                    var1[icount-1] = float(var1[icount-1])
                elif parm.apex_var[i]==4:           #Mineral N,kg
                    var1[icount-1] = float(rch_data[19])+float(rch_data[21])+float(rch_data[23]) 
                    var1[icount-1] = float(var1[icount-1]) 
                elif parm.apex_var[i]==5:           #Organic N,kg
                    var1[icount-1] = float(rch_data[15]) 
                elif parm.apex_var[i]==6:           #Mineral p,kg
                    var1[icount-1] = float(rch_data[25]) 
                elif parm.apex_var[i]==7:           #Organic p,kg
                    var1[icount-1] = float(rch_data[17]) 
                elif parm.apex_var[i]==8:           #Pesticide,grams
                    var1[icount-1] = float(rch_data[33])+float(rch_data[35]) 
                    var1[icount-1] = float(var1[icount-1]) 
            if icount==len(parm.pred_datea[i]):
                break

        parm.pred_datea[i] = parm.pred_date[i] 
        parm.pred_val[i] = var1[:]
