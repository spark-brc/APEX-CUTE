import parm, math
import numpy as np

#def performance_indicators(pred,obs):
#    diff = np.array(obs) - np.array(pred)
#    meanob = np.mean(obs)
#    parm.meanpr = np.mean(pred)
#    parm.stdpr = np.std(pred)
#    parm.re = (meanob - parm.meanpr) / meanob * 100 #Percent error PBIAS   
#    parm.bias = np.sum(np.fabs(np.array(diff))) / len(diff) #AD
#    parm.rmse = math.sqrt(sum(np.array(diff)**2) / len(diff))

#    parm.r2 = np.corrcoef(obs,pred)[0,1]**2    
#    s = np.sum((np.array(obs) - meanob) ** 2)            
#    parm.nse = 1 - np.sum(np.array(diff) ** 2) / s

def performance_indicators(pred,obs):
    diff = list(np.array(obs) - np.array(pred))
    meanob = np.mean(obs)
    parm.meanpr = np.mean(pred)
    parm.stdpr = np.std(pred)
    parm.re = (meanob - parm.meanpr) / meanob * 100 #Percent error PBIAS   
    parm.bias = np.sum(np.fabs(np.array(diff))) / len(diff) #AD
    parm.rmse = math.sqrt(sum(np.array(diff)**2) / len(diff))

    parm.r2 = np.corrcoef(obs,pred)[0,1]**2    
    s = np.sum((np.array(obs) - meanob) ** 2)            
    parm.nse = 1 - np.sum(np.array(diff) ** 2) / s