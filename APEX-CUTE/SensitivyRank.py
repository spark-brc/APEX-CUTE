import parm, os, csv,math
import msgbox
        
def read(si_method):
    parm.si_parm=[]
    parm.si_first=[]
    parm.si_total=[]

    if si_method == 'Sobol':
        fname = parm.path_proj + '\\Sensitivity_Rank_Sobol.csv'   
    else:
        fname = parm.path_proj + '\\Sensitivity_Rank_FAST.csv'   

    if not os.path.isfile(fname):
        parm.error_msg ="An error occured while reading \n"+fname
        msgbox.msg("Error ", parm.error_msg)
        parm.iflg=1; return

    with open(fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                parm.si_parm.append(row[0])
                parm.si_first.append(abs(float(row[1])))
                parm.si_total.append(abs(float(row[2])))
                line_count += 1
            else:
                line_count += 1

        