import os, parm
import numpy as np
import msgbox


#this module updates PRMT values in PARM.DAT

#Parameter list 
plist =    ('PARM1' ,	'PARM2' ,	'PARM3' ,	'PARM4' ,	'PARM5' ,	'PARM6' ,	'PARM7' ,	'PARM8' ,	'PARM9' ,	'PARM10',
            'PARM11',	'PARM12',	'PARM13',	'PARM14',	'PARM15',	'PARM16',	'PARM17',	'PARM18',	'PARM19',	'PARM20',
            'PARM21',	'PARM22',	'PARM23',	'PARM24',	'PARM25',	'PARM26',	'PARM27',	'PARM28',	'PARM29',	'PARM30',
            'PARM31',	'PARM32',	'PARM33',	'PARM34',	'PARM35',	'PARM36',	'PARM37',	'PARM38',	'PARM39',	'PARM40',
            'PARM41',	'PARM42',	'PARM43',	'PARM44',	'PARM45',	'PARM46',	'PARM47',	'PARM48',	'PARM49',	'PARM50',
            'PARM51',	'PARM52',	'PARM53',	'PARM54',	'PARM55',	'PARM56',	'PARM57',	'PARM58',	'PARM59',	'PARM60',
            'PARM61',	'PARM62',	'PARM63',	'PARM64',	'PARM65',	'PARM66',	'PARM67',	'PARM68',	'PARM69',	'PARM70',
            'PARM71',	'PARM72',	'PARM73',	'PARM74',	'PARM75',	'PARM76',	'PARM77',	'PARM78',	'PARM79',	'PARM80',
            'PARM81',	'PARM82',	'PARM83',	'PARM84',	'PARM85',	'PARM86',	'PARM87',	'PARM88',	'PARM89',	'PARM90',
            'PARM91',	'PARM92',	'PARM93',	'PARM94',	'PARM95',	'PARM96',	'PARM97',	'PARM98',	'PARM99',   'PARM100',
            'PARM101',	'PARM102',	'PARM103',	'PARM104',	'PARM105',	'PARM106',	'PARM107',	'PARM108',	'PARM109',  'PARM110')

nparm = len(plist)
prmt = ['' for x in range(0,nparm+1)]

def update():

    #This module updates APEX's PARM file with new set of parameters

    fnam = parm.path_TxtWork + '/' + parm.fnam_parm
    try:
        parm.fnam1 = open(fnam, 'r+')
    except:
        #Print error message and exit
        parm.error_msg = fnam + ' is not found.'
        msgbox.msg("Error", parm.error_msg)
        parm.iflg=1; return
   
    #Creat a temporary file
    try:
         parm.fnam2 = open(parm.path_TxtWork + '/' +'temp.tmp','w') #temp file
    except:
        #Print error message and exit
        parm.error_msg = "An error was occurred while \n creating temp.tmp for APEX PARM file."
        msgbox.msg("Error", parm.error_msg)
        parm.iflg=1; return


    linestr= parm.fnam1.readlines()

    #read parm values from parm.dat
    for i in range (0,11):
        txtline = linestr[i+35]
        dum = ' '.join(txtline.split())   #updated for APEX1501 Jan 2017 
        data = dum.split(' ')
        if len(data)<10:
            data = list(chunkstring(txtline,8))   #updated for APEX1501 May 2017 
        for j in range (0,len(data)):
            k = i * 10 + j +1
            prmt[k] = data[j]
            if prmt[k] == '        ': prmt[k] = '0.'
            if k == nparm : break

    #update values for selected parms
    for j in range(len(parm.par_name)):
        if parm.par_name[j][0:4].upper() == 'PARM':
            for k in range(0,nparm):
               if parm.par_name[j].upper() == plist[k]:
                   txt = str("{:8.4f}".format(parm.cur_test_var[j]))
                   if len(txt)>8:
                       txt = txt[len(txt)-8:len(txt)]
                   prmt[k+1] = txt
                   break
   
    #update the line 
    for i in range(0,11):
        txt = ''
        for j in range(0,10):
            k = i * 10 + j+1
            txt = txt + str("{0:>8}".format(prmt[k]))   

        txt = txt + '\n'
        linestr[i+35] = txt

    #reprint PARM.DAT       
    parm.fnam2.writelines(linestr[0:len(linestr)]) 
    
    parm.fnam1.close()
    parm.fnam2.close()
    os.remove(fnam)
    os.rename(parm.path_TxtWork + '\\temp.tmp',fnam)

def chunkstring(string, length):
    return (string[0+ii:length+ii] for ii in range(0, len(string), length))
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% End of Amir's additions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




