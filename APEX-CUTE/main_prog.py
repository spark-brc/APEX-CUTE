import sys, os, os.path, parm, ctypes, shutil, time, datetime, math, random
import PyQt5, subprocess
from PyQt5 import QtWidgets, QtCore, uic, QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import numpy as np
import File_toUpdate, msgbox, data_pairing
from dateutil.parser import parse

import matplotlib
matplotlib.use('QT5Agg')

import matplotlib.pylab as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas 
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

parm.cute_rev = 'APEX-CUTE7.6'
class MyWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
    
        super(MyWindow, self).__init__()
        uic.loadUi('cute_gui_r1.ui', self)
        self.setWindowTitle(parm.cute_rev)
        self.setWindowIcon(QtGui.QIcon('ApexCUTE2.png'))
        # dummy plot
        labels = ['P1','P2','P3','P4','P5','P6']
        x = np.arange(len(labels))
        width = 0.35
        data = np.array([0.2,0.7,0.5,0.1,0.9,0.3])        
        self.fig, ax1 = plt.subplots()
        rect = ax1.bar(x-width/2, data,width)
        ax1.set_ylabel('Sensitivity Index')
        ax1.set_title('Total Sensitivity of Parameters')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        self.canvas = FigureCanvas(self.fig)
        lay = QtWidgets.QVBoxLayout(self.MplWidget)  
        lay.setContentsMargins(0, 0, 0, 0)      
        lay.addWidget(self.canvas)
        # add toolbar
        self.toolbar = NavigationToolbar(self.canvas,self.MplWidget)
        lay.addWidget(self.toolbar)

        self.show()

        #load saved path if exists
        parm.cute_path = str(os.getcwd())
        if os.path.isfile('path.dat')==True:
            if os.stat('path.dat')[6]>0:
                fnam = open('path.dat', 'r')
                for txtline in fnam:
                    parm.path_proj = txtline
                    parm.path_apex = txtline + "\TxtInOut"
                fnam.close()
            else:
                parm.path_proj.setText(str(os.getcwd()))
                os.remove('path.dat')
        else:
            parm.path_proj = str(os.getcwd())

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        #self.statusBar.showMessage('Texas A&M AgriLife Research')

        #toolbar menu
        self.action_New_Project.triggered.connect(new_cute_proj)
        self.action_Open_Project.triggered.connect(open_cute_proj)
        self.actionSave_Project.triggered.connect(save_proj)
        self.actionSave_As.triggered.connect(saveas_proj)
        self.action_Quit.triggered.connect(close_proj)
        self.actionQuit_APEX_CUTE.triggered.connect(sigint_handler)


        #tab: Main
        self.btn_newproj.clicked.connect(new_cute_proj)
        self.btn_openproj.clicked.connect(open_cute_proj)
        #self.btn_saveproj.clicked.connect(save_proj)
        #self.btn_saveprojas.clicked.connect(saveas_proj)
        self.brow_apex.clicked.connect(browse_apex_folder)
        self.btn_apply_path.clicked.connect(set_path)
        self.btn_cal_sa.clicked.connect(reset_config_status)
        self.btn_cal_setting.clicked.connect(check_cal_setting)

        #tab: Outlet Setting
        self.btn_cute_setting_save.clicked.connect(cute_setting_save)
        self.txt_apex_out_1.currentIndexChanged.connect(self.crop_selected1)
        self.txt_apex_out_2.currentIndexChanged.connect(self.crop_selected2)
        self.txt_apex_out_3.currentIndexChanged.connect(self.crop_selected3)
        self.txt_apex_out_4.currentIndexChanged.connect(self.crop_selected4)

        #tab: PARMs
        self.btn_parm_save.clicked.connect(read_PARMS)
        self.btn_filter.clicked.connect(filter_parms)
        self.btn_clear.clicked.connect(filter_parms_clear)
        #tab: Other parameters
        self.btn_apexparm_save.clicked.connect(read_inputparameters)
        #tab: Crop Parameters
        self.btn_cropparm_save.clicked.connect(read_CROP)
        #tab: Constraints
        self.btn_constraints.clicked.connect(read_constraints)
        #tab: Observed data
        self.btn_obsexample.clicked.connect(open_obs_example)
        self.btn_user_obs_file_path.clicked.connect(browse_obs_file)
        self.btn_user_obs_open.clicked.connect(open_obs_file)
        self.btn_save_user_obs.clicked.connect(save_obs_file) 

        #Run 
        self.btn_save.clicked.connect(btn_confirm)
        self.btn_run.clicked.connect(btn_run)  #Run APEX-CUTE
        self.btn_plot.clicked.connect(self.update_graph)
        self.rb_parm_sensitivity.toggled.connect(self.btnselected)
        self.btn_log.clicked.connect(save_log_file) 
 
        self.tabWidget.currentChanged.connect(self.onChange)

    #disable crop name in the Outlets tab if crop name is not needed for selected APEX output
    def crop_selected1(self,id):
        if id>=24 and id<29: ui.txt_crop_1.setEnabled(True)
        else: ui.txt_crop_1.setDisabled(True)
    def crop_selected2(self,id):
        if id>=24 and id<29: ui.txt_crop_2.setEnabled(True)
        else: ui.txt_crop_2.setDisabled(True)
    def crop_selected3(self,id):
        if id>=24 and id<29: ui.txt_crop_3.setEnabled(True)
        else: ui.txt_crop_3.setDisabled(True)
    def crop_selected4(self,id):
        if id>=24 and id<29: ui.txt_crop_4.setEnabled(True)
        else: ui.txt_crop_4.setDisabled(True)

    def btnselected(self, value):
        #get the list of calibrated parameters and add to the dropdown combobox
        rbtn = self.sender()
        if rbtn.isChecked() == True:
                #read the list of calibrated parameters.
                fname = parm.path_proj + '\\dds.out'   
                try:
                    fnam = open(fname, 'r+')
                except:
                    return
                for txtline in fnam:
                    txtline = txtline.split()
                    parm.dds_head = txtline
                    break 
                fnam.close()  
                pnum = len(parm.dds_head) - 3
                if pnum>0:
                    self.combo_parm.clear()
                    for i in range(1,pnum+1):
                        self.combo_parm.addItem(parm.dds_head[i])
        
    def onChange(self,i):
        pnum = 0
        obid1 = ''
        obid2 = 1
        #sync obs data settings in "Observed data" tab to "Outlets" tab
        self.progressBar_1.setValue(0)
        if i==7: # tab "Obs"
            if self.txt_sub_1.text()!='': self.txt_user_obs_subnum.setText(self.txt_sub_1.text())
            if parm.path_proj!='': self.txt_user_obs_save_path.setText(parm.path_proj+'\Obs')

            if self.txt_dt_1.currentText().upper()=='MONTHLY': 
                self.rb_user_obs_mon.setChecked(True)
                obid1='M'
            elif self.txt_dt_1.currentText().upper()=='DAILY': 
                self.rb_user_obs_day.setChecked(True)
                obid1='D'
            elif self.txt_dt_1.currentText().upper()=='YEARLY': 
                self.rb_user_obs_yr.setChecked(True)
                obid1='Y'

            if self.txt_apex_out_1.currentIndex()<9: #RCH 
                self.rb_user_obs_type_rch.setChecked(True)
                obid2=0
            elif self.txt_apex_out_1.currentIndex()<28: #SUB
                self.rb_user_obs_type_sub.setChecked(True)
                obid2=1
            else: #GRAIN YIELD (ACY)
                self.rb_user_obs_type_crp.setChecked(True)
                obid2=2

            if obid2==0: #RCH
                if obid1=='M': self.drop_obsexamp.setCurrentIndex(0)
                if obid1=='D': self.drop_obsexamp.setCurrentIndex(1)
                if obid1=='Y': self.drop_obsexamp.setCurrentIndex(2)
            elif obid2==1: #SUB
                if obid1=='M': self.drop_obsexamp.setCurrentIndex(3)
                if obid1=='D': self.drop_obsexamp.setCurrentIndex(4)
                if obid1=='Y': self.drop_obsexamp.setCurrentIndex(5)
            else:
                self.drop_obsexamp.setCurrentIndex(6)

        elif i==8: # tab "Run"
            if parm.cute_option=='Calib':
                ui.rb_of.setChecked(True)
            elif parm.cute_option=='Sobol':
                ui.rb_sobol_si.setChecked(True)
            else:
                ui.rb_fast_si.setChecked(True)
            
            #read the list of calibrated parameters.
            fname = parm.path_proj + '\\dds.out'   
            try:
                fnam = open(fname, 'r+')
            except:
                return
            for txtline in fnam:
                txtline = txtline.split()
                parm.dds_head = txtline
                break 
            fnam.close()  
            pnum = len(parm.dds_head) - 3
            if pnum>0:
                self.combo_parm.clear()
                for i in range(1,pnum+1):
                    self.combo_parm.addItem(parm.dds_head[i])

    def update_graph(self):
        import SensitivyRank, read_ddsout
        ylabel=''
        pname=''

        if ui.rb_sobol_si.isChecked():
             SensitivyRank.read('Sobol') #read sobol sensitivity index rank output.
          
        elif ui.rb_fast_si.isChecked():
             SensitivyRank.read('FAST') #read fast sensitivity index rank output.

        if parm.iflg==1: return

        if ui.rb_sobol_si.isChecked() or ui.rb_fast_si.isChecked():
            data=[]
            if ui.cb_sensitivity_rank.currentText()=='Global':
                ylabel = 'Total Sensitivity'
                data = parm.si_total
            else:
                ylabel = 'First-order Sensitivity'
                data = parm.si_first

            self.fig.clf()
            ax = self.fig.add_subplot(111)
            x = np.arange(len(parm.si_parm))
            width = 0.35
            rect = ax.bar(x-width/2, data,width)
            ax.set_ylabel(ylabel)
            
            ax.set_title('Sensitivity Index')
            ax.set_xticks(x)
            ax.set_xticklabels(parm.si_parm)
            self.canvas.draw()

        elif ui.rb_of.isChecked():
            data=[]
            #read dds.out
            read_ddsout.read()
            if parm.iflg==1: return

            cnum = len(parm.dds_head) # columns
            rnum = len(parm.dds_data) # rows
            for i in range(rnum): 
                data.append(parm.dds_data[i,cnum-1])
            self.fig.clf()
            ax = self.fig.add_subplot(111)
            ax.set_ylabel('Objective Function')
            ax.set_title('Optimization Curve')
            ax.plot(data)
            self.canvas.draw()

        elif ui.rb_uncertainty.isChecked():
            a=a
        else:
            xval=[]
            yval=[]
            read_ddsout.read()
            if parm.iflg==1: return

            rnum = len(parm.dds_data) # rows
            cnum = len(parm.dds_head)-1 # columns
            pname = str(self.combo_parm.currentText())

            for i in range(cnum): 
                if parm.dds_head[i]==pname: break

            for j in range(rnum):
                xval.append(parm.dds_data[j,i])
                yval.append(parm.dds_data[j,cnum-1])

            self.fig.clf()
            ax = self.fig.add_subplot(111)
            ax.set_xlabel(pname)
            ax.set_ylabel('Objective Function')
            ax.set_title(pname)
            ax.plot(xval,yval,'o')
            self.canvas.draw()
        
def close_proj():
    if parm.proj_cur==1:
            msgbox.msg("Notice", "Current project will be saved before closing.")
            save_proj()
            parm.file_proj = ''
            parm.path_proj = ''
            parm.path_proj = ''
            ui.setWindowTitle(parm.cute_rev)     

            filter_parms_clear()

            allRows = ui.tbl_crop_parameters.rowCount()
            for row in range(0,allRows):
               ui.tbl_crop_parameters.setItem(row,1, QTableWidgetItem("0"))

            allRows = ui.tbl_constraints.rowCount()
            for row in range(0,allRows):
                ui.tbl_constraints.setItem(row,1, QTableWidgetItem("0"))
    else:
        msgbox.msg("Notice", "There is no current project to close!")
def filter_parms():
    allRows = ui.tbl_parms.rowCount()
    for j in range(0,allRows):
        ui.tbl_parms.setItem(j, 1, QTableWidgetItem("0"))

    for j in range(0,allRows):
        if ui.chk_flow.isChecked():
            if parm.parm_type[j]==1:
                ui.tbl_parms.setItem(j, 1, QTableWidgetItem("1"))
        if ui.chk_sed.isChecked():
            if parm.parm_type[j]==2:
                ui.tbl_parms.setItem(j, 1, QTableWidgetItem("1"))
        if ui.chk_n.isChecked():
            if parm.parm_type[j]==3:
                ui.tbl_parms.setItem(j, 1, QTableWidgetItem("1"))
        if ui.chk_p.isChecked():
            if parm.parm_type[j]==4:
                ui.tbl_parms.setItem(j, 1, QTableWidgetItem("1"))
        if ui.chk_carbon.isChecked():
            if parm.parm_type[j]==5:
                ui.tbl_parms.setItem(j, 1, QTableWidgetItem("1"))
        if ui.chk_pest.isChecked():
            if parm.parm_type[j]==6:
                ui.tbl_parms.setItem(j, 1, QTableWidgetItem("1"))
        if ui.chk_crop.isChecked():
            if parm.parm_type[j]==7: 
                ui.tbl_parms.setItem(j, 1, QTableWidgetItem("1"))

def filter_parms_clear():
    allRows = ui.tbl_parms.rowCount()
    
    for j in range(0,allRows):
        ui.tbl_parms.setItem(j, 1, QTableWidgetItem("0"))

        ui.chk_flow.setChecked(False)
        ui.chk_sed.setChecked(False)
        ui.chk_n.setChecked(False)
        ui.chk_p.setChecked(False)
        ui.chk_carbon.setChecked(False)
        ui.chk_pest.setChecked(False)
        ui.chk_crop.setChecked(False)

def browse_obs_file():
    #browse observed data file
    fobs=''
    if os.path.isdir(parm.path_proj)==True:
        fobs=parm.path_proj+'\Obs'
    else:
        fobs=parm.cute_path+'\project1501\Obs'
    str='Open a data file'
    fobs = QFileDialog.getOpenFileName(None,str,fobs,"Observed Data File (*.csv)")
    fileobs = os.path.normpath(fobs[0])
    ui.txt_user_obs_path.setText(fileobs)

def open_obs_file():
    #open user obs file and show in the textbox
    fob=''
    fobs=ui.txt_user_obs_path.toPlainText()
    if os.path.isfile(fobs)==False or os.stat(fobs).st_size == 0:
        msgbox.msg("Error", "The file does not exist or is empty. \n Choose a different file.")
        return
    
    str_array=[]
    with open(fobs) as f:
        str_array = f.read().splitlines() 
    f.close()
    ui.txt_obs_show.clear()
    for i in range(len(str_array)):
        ui.txt_obs_show.append(str_array[i])

def save_obs_file():
    fnam=''
    fpath=ui.txt_user_obs_save_path.toPlainText()
    if os.path.isdir(fpath)==False:
        msgbox.msg("Error", "The folder does not exis in \n"+" "+fpath)
        return
    if str(ui.txt_obs_show.toPlainText())=='':
        msgbox.msg("Error", "No data is entered.")
        return
    
    if ui.rb_user_obs_type_crp.isChecked():
        fnam = fpath + '\\obs_crop.csv'
    elif ui.rb_user_obs_type_rch.isChecked():
        fnam = fpath + '\\rch_'
    else:
        fnam = fpath + '\\sub_'
       
    if ui.rb_user_obs_yr.isChecked():
        fnam = fnam + 'yearly' + ui.txt_user_obs_subnum.text() + '.csv'
    elif ui.rb_user_obs_day.isChecked():
        fnam = fnam + 'daily' + ui.txt_user_obs_subnum.text() + '.csv'
    else:
        fnam = fnam + 'monthly' + ui.txt_user_obs_subnum.text() + '.csv'

    linestr = ui.txt_obs_show.toPlainText().split('\n')
    file1=open(fnam, 'w+')
    for i in range(len(linestr)): file1.write(linestr[i]+'\n')
    file1.close()
    
    ui.messages.append('An obs file is saved to:')
    ui.messages.append(fnam + '\n')

def open_obs_example():
    fnam=''
    fpath=''
    fpath='C:\APEX\APEX-CUTE\project1501\Obs'
    if os.path.isdir(fpath)==False:
        fpath=parm.cute_path+'\\project1501\\Obs'
        if os.path.isdir(fpath)==False:
            parm.error_msg = "Example observation data is not found.\n"
            msgbox.msg("Error", parm.error_msg)
            ui.messages.append("Error: "+parm.error_msg)
            parm.iflg=1; return

    if ui.drop_obsexamp.currentText()=='RCH-monthly':
        fnam=fpath+'\\rch_monthly2.csv'
    elif ui.drop_obsexamp.currentText()=='RCH-daily':
        fnam=fpath+'\\rch_daily2.csv'
    elif ui.drop_obsexamp.currentText()=='RCH-yearly':
        fnam=fpath+'\\rch_yearly2.csv'
    elif ui.drop_obsexamp.currentText()=='SUB-monthly':
        fnam=fpath+'\\sub_monthly1.csv'
    elif ui.drop_obsexamp.currentText()=='SUB-daily':
        fnam=fpath+'\\sub_daily1.csv'
    elif ui.drop_obsexamp.currentText()=='SUB-yearly':
        fnam=fpath+'\\sub_yearly1.csv'
    else:
        fnam=fpath+'\\obs_crop.csv'

    if os.path.isfile(fnam)==False or os.stat(fnam).st_size == 0:
        parm.error_msg = "The example obs file does not exis in \n"+" "+fpath
        msgbox.msg("Error", parm.error_msg)
        ui.messages.append("Error: "+parm.error_msg)
        parm.iflg=1; return

    str_array=[]
    with open(fnam) as f:
        str_array = f.read().splitlines() 
    f.close()
    ui.txt_obs_show.clear()
    for i in range(len(str_array)):
        ui.txt_obs_show.append(str_array[i])

def new_cute_proj():
    str = 'New CUTE project'
    nameproj = QFileDialog.getSaveFileName(None,caption=str,directory=parm.path_proj,filter="Project File (*.cproj)")
    parm.file_proj = os.path.abspath(nameproj[0])
    parm.path_proj = os.path.dirname(nameproj[0])
    parm.path_proj = os.path.normpath(parm.path_proj)
    
    if os.path.isdir(parm.path_proj)==True:
        ui.setWindowTitle(parm.file_proj)
        try:
            fnam = open(parm.file_proj, 'w')
        except IOError:
            return
        fnam.write(parm.path_proj  + '\n')
        fnam.close()
        parm.proj_cur=1 
        ui.messages.append('New project is created: ')
        ui.messages.append(parm.file_proj  + '\n')

    else:
        ui.messages.append('The path does not exist. Try again with a valid path.')
        ui.messages.append(nameproj  + '\n')
        parm.proj_cur=0 

def browse_apex_folder():
    #open folder browser
    str = 'Browse to the APEX folder'
    destDir = QFileDialog.getExistingDirectory(None, str, parm.path_proj, QFileDialog.ShowDirsOnly)
    destDir = os.path.normpath(destDir)

    if os.path.isdir(destDir)==True:
        ui.path_apex.setText(destDir)
        ui.messages.append('APEX files will be copied to the project folder from: ')
        ui.messages.append(destDir + '\n')
        parm.path_apex = ui.path_apex.text()

def open_cute_proj():
    parm.apex_outlets=[]
    parm.obs_dt=[]
    parm.of_weight=[]
    parm.apex_var=[]
    parm.apex_output=[]
    parm.apex_crop
    parm.iflg=0

    #open cute project file *.cproj
    str = 'Open a CUTE project'
    nameproj = QFileDialog.getOpenFileName(None,str,parm.path_proj,"Project File (*.cproj)")

    if os.path.isfile(nameproj[0])==False or os.stat(nameproj[0]).st_size == 0:
        msgbox.msg("Notice", "No project file is selected. \n Choose a file.")
        parm.iflg=1;parm.proj_cur=0;return

    parm.file_proj = nameproj[0]
    parm.path_proj = os.path.dirname(nameproj[0])
    parm.path_proj = os.path.normpath(parm.path_proj)
    parm.file_proj = os.path.normpath(parm.file_proj)

    parm.path_TxtInout = parm.path_proj + '\TxtInOut'
    parm.path_TxtWork = parm.path_proj + '\TxtWork'
    parm.path_obs = parm.path_proj + '\Obs'
    parm.iflg=0

    ui.setWindowTitle(parm.file_proj)
    parm.proj_cur=1
    proj_array = []
 
    #Read project file
    with open(parm.file_proj) as f:
        proj_array = f.read().splitlines() 
    f.close()

    ##line 1 - Main tab
    parm.path_apex = proj_array[0] 
    ui.path_apex.setText(parm.path_apex)

    #line 2 - Task tab
    parm.cute_option = proj_array[1] 
    if parm.cute_option.upper() == 'CALIB':
        ui.rb_calibration.setChecked(True)
    else:
        ui.rb_sa.setChecked(True)
        if parm.cute_option.upper() == 'SOBOL':
            ui.rb_sa_sobol.setChecked(True)
        elif parm.cute_option.upper() =='FAST':
            ui.rb_sa_fast.setChecked(True)
        else:
            msgbox.msg("Error", "An error occurred while reading Line 2. \n"+"Option: Calib/Sobol/Fast.")
            parm.iflg=1;parm.proj_cur=0;return

    #line 3
    if proj_array[2].isnumeric() == True:
        parm.SA_n = int(proj_array[2]) 
    else:
        msgbox.msg("Error", "An error occurred while reading Line 3. \n"+"Enter a positive whole number.")
        parm.iflg=1;parm.proj_cur=0;return
    ui.txt_num_sa_interval.setText(proj_array[2])

    #Line 4
    txts = proj_array[3].split(',')   
    if parse(txts[0]) == False or parse(txts[1]) == False:
        msgbox.msg("Error", "An error occurred while reading Line 4. \n"+"Option: two dates in M/d/yyyy separated by a comma.")
        parm.iflg=1;parm.proj_cur=0;return

    dat=QDate.fromString(txts[0],'M/d/yyyy')
    ui.txt_calibration_start_date.setDate(dat)
    parm.start_cal = datetime.date(QDate.year(dat),QDate.month(dat),QDate.day(dat))

    dat=QDate.fromString(txts[1],'M/d/yyyy')
    ui.txt_calibration_end_date.setDate(dat)
    parm.end_cal = datetime.date(QDate.year(dat),QDate.month(dat),QDate.day(dat))
    if parm.start_cal > parm.end_cal:
        msgbox.msg("Error", "An error occurred while reading Line 4. \n"+"Starting date of calibration is later than the ending date.")
        parm.iflg=1;parm.proj_cur=0;return
   
    #line 5
    txts = proj_array[4].split(',')   
    if parse(txts[0]) == False or parse(txts[1]) == False:
        msgbox.msg("Error", "An error occurred while reading Line 5. \n"+"Option: two dates in M/d/yyyy separated by a comma.")
        parm.iflg=1;parm.proj_cur=0;return

    dat=QDate.fromString(txts[0],'M/d/yyyy')
    ui.txt_validation_start_date.setDate(dat)
    parm.start_val = datetime.date(QDate.year(dat),QDate.month(dat),QDate.day(dat))
    
    dat=QDate.fromString(txts[1],'M/d/yyyy')
    ui.txt_validation_end_date.setDate(dat)
    parm.end_val = datetime.date(QDate.year(dat),QDate.month(dat),QDate.day(dat))
    if parm.start_val > parm.end_val:
        msgbox.msg("Error", "An error occurred while reading Line 5. \n"+"Starting date of validation is later than the ending date.")
        parm.iflg=1;parm.proj_cur=0;return

    #line 6
    if proj_array[5].isnumeric() == True:
       parm.dds_ndraw = int(proj_array[5]) 
    else:
        msgbox.msg("Error", "An error occurred while reading Line 5. \n"+"Enter a positive whole number.")
        parm.iflg=1;parm.proj_cur=0;return
    ui.txt_dds_total_num.setText(proj_array[5])

    #line 7
    parm.dds_stat = proj_array[6]
    if parm.dds_stat.upper()=="F(NSE-PBIAS)":
        ui.txt_stat.setCurrentIndex(1)
    elif parm.dds_stat.upper()=="R2":
        ui.txt_stat.setCurrentIndex(2)
    elif parm.dds_stat.upper()=="RMSE":
        ui.txt_stat.setCurrentIndex(3)
    elif parm.dds_stat.upper()=="ABSOLUTE ERROR":
        ui.txt_stat.setCurrentIndex(4)
    elif parm.dds_stat.upper()=="PBIAS":
        ui.txt_stat.setCurrentIndex(5)
    elif parm.dds_stat.upper()=="NSE":
        ui.txt_stat.setCurrentIndex(0)
    else:
        msgbox.msg("Error", "An error occurred while reading Line 7. \n"+"Selected Performance indicator is incorrect.")
        ui.progressBar.setValue(0)        
        parm.iflg=1;parm.proj_cur=0;return

    #line 8
    if proj_array[7].upper()=='NEW CALIBRATION': #new or continuing calibration?
        parm.dds_restart = 0 
        ui.drop_calib_option.setCurrentIndex(0)
    elif proj_array[7].upper()=='CONTINUE THE PREVIOUS RUN':
        parm.dds_restart = 1
        ui.drop_calib_option.setCurrentIndex(1)
    else:
        msgbox.msg("Error", "An error occurred while reading Line 8. \n"+"Option: New calibration/Continue the previous run.")
        parm.iflg=1;parm.proj_cur=0;return

    #line 9
    if proj_array[8].upper()=='USER DEFAULT VALUES': #initial parameters for DDS:  or random search
        parm.dds_useinit = 1  #use user input
        ui.drop_dds_init_cond.setCurrentIndex(0)
    elif proj_array[8].upper()=='RANDOM SAMPLING':
        parm.dds_useinit = 0 #use random sampling
        ui.drop_dds_init_cond.setCurrentIndex(1)
    else:
        msgbox.msg("Error", "An error occurred while reading Line 9. \n"+"Option: User default values/Random sampling.")
        parm.iflg=1;parm.proj_cur=0;return

    ui.progressBar.setValue(25)
    qApp.processEvents()
    time.sleep(0.1)


    #line 10 - Outlets tab
    txts = proj_array[9].split(',') 
    if txts[0].isnumeric() == False or txts[1].isnumeric() == True :
        msgbox.msg("Error", "An error occurred while reading Line 10. \n"+"Outlets tab's 1st column inputs incorrect.")
        ui.progressBar.setValue(0)
        parm.iflg=1;parm.proj_cur=0;return

    ui.txt_sub_1.setText(txts[0])
    parm.apex_outlets.append(int(txts[0]))
    if txts[1].upper()=="MONTHLY":
        parm.obs_dt.append("MONTHLY")
        ui.txt_dt_1.setCurrentIndex(0)
    elif txts[1].upper()=="DAILY":
        parm.obs_dt.append("DAILY")
        ui.txt_dt_1.setCurrentIndex(1)
    elif txts[1].upper()=="YEARLY":
        parm.obs_dt.append("YEARLY")
        ui.txt_dt_1.setCurrentIndex(2)
    else:
        parm.obs_dt.append("YEARLY")
        ui.txt_dt_1.setCurrentIndex(3)

    if txts[2].upper()=="RCH-FLOW":
        parm.apex_var.append(0)
        ui.txt_apex_out_1.setCurrentIndex(0)
    elif txts[2].upper()=="RCH-SEDIMENT":
        parm.apex_var.append(1)
        ui.txt_apex_out_1.setCurrentIndex(1)
    elif txts[2].upper()=="RCH-TN":
        parm.apex_var.append(2)
        ui.txt_apex_out_1.setCurrentIndex(2)
    elif txts[2].upper()=="RCH-TP":
        parm.apex_var.append(3)
        ui.txt_apex_out_1.setCurrentIndex(3)
    elif txts[2].upper()=="RCH-MINERALN":
        parm.apex_var.append(4)
        ui.txt_apex_out_1.setCurrentIndex(4)
    elif txts[2].upper()=="RCH-ORGANICN":
        parm.apex_var.append(5)
        ui.txt_apex_out_1.setCurrentIndex(5)
    elif txts[2].upper()=="RCH-MINERALP":
        parm.apex_var.append(6)
        ui.txt_apex_out_1.setCurrentIndex(6)
    elif txts[2].upper()=="RCH-ORGANICP":
        parm.apex_var.append(7)
        ui.txt_apex_out_1.setCurrentIndex(7)
    elif txts[2].upper()=="RCH-PEST":
        parm.apex_var.append(8)
        ui.txt_apex_out_1.setCurrentIndex(8)
    elif txts[2].upper()=="SUB-WATERYLD":
        parm.apex_var.append(9)
        ui.txt_apex_out_1.setCurrentIndex(9)
    elif txts[2].upper()=="SUB-RUNOFF":
        parm.apex_var.append(10)
        ui.txt_apex_out_1.setCurrentIndex(10)
    elif txts[2].upper()=="SUB-QDR":
        parm.apex_var.append(11)
        ui.txt_apex_out_1.setCurrentIndex(11)
    elif txts[2].upper()=="SUB-RESQ":
        parm.apex_var.append(12)
        ui.txt_apex_out_1.setCurrentIndex(12)
    elif txts[2].upper()=="SUB-PET":
        parm.apex_var.append(13)
        ui.txt_apex_out_1.setCurrentIndex(13)
    elif txts[2].upper()=="SUB-ET":
        parm.apex_var.append(14)
        ui.txt_apex_out_1.setCurrentIndex(14)
    elif txts[2].upper()=="SUB-SW":
        parm.apex_var.append(15)
        ui.txt_apex_out_1.setCurrentIndex(15)
    elif txts[2].upper()=="SUB-SEDYLD":
        parm.apex_var.append(16)
        ui.txt_apex_out_1.setCurrentIndex(16)
    elif txts[2].upper()=="SUB-QN":
        parm.apex_var.append(17)
        ui.txt_apex_out_1.setCurrentIndex(17)
    elif txts[2].upper()=="SUB-QP":
        parm.apex_var.append(18)
        ui.txt_apex_out_1.setCurrentIndex(18)
    elif txts[2].upper()=="SUB-ORGN":
        parm.apex_var.append(19)
        ui.txt_apex_out_1.setCurrentIndex(19)
    elif txts[2].upper()=="SUB-ORGP":
        parm.apex_var.append(20)
        ui.txt_apex_out_1.setCurrentIndex(20)
    elif txts[2].upper()=="SUB-QDRN":
        parm.apex_var.append(21)
        ui.txt_apex_out_1.setCurrentIndex(21)
    elif txts[2].upper()=="SUB-QDRP":
        parm.apex_var.append(22)
        ui.txt_apex_out_1.setCurrentIndex(22)
    elif txts[2].upper()=="SUB-STMP":
        parm.apex_var.append(23)
        ui.txt_apex_out_1.setCurrentIndex(23)
    elif txts[2].upper()=="CRP-BIOM":
        parm.apex_var.append(24)
        ui.txt_apex_out_1.setCurrentIndex(24)
    elif txts[2].upper()=="CRP-STL":
        parm.apex_var.append(25)
        ui.txt_apex_out_1.setCurrentIndex(25)
    elif txts[2].upper()=="CRP-LAI":
        parm.apex_var.append(26)
        ui.txt_apex_out_1.setCurrentIndex(26)
    elif txts[2].upper()=="CRP-CHT":
        parm.apex_var.append(27)
        ui.txt_apex_out_1.setCurrentIndex(27)
    elif txts[2].upper()=="ACY-YLDG":
        parm.apex_var.append(28)
        ui.txt_apex_out_1.setCurrentIndex(28)
    elif txts[2].upper()=="SED-HFLUX":
        parm.apex_var.append(29)
        ui.txt_apex_out_1.setCurrentIndex(29)
    elif txts[2].upper()=="SED-VFLUX":
        parm.apex_var.append(30)
        ui.txt_apex_out_1.setCurrentIndex(30)
    else:
        msgbox.msg("Error", "An error occurred while reading Line 10. \n"+"Outlets tab's 1st column inputs incorrect.")
        ui.progressBar.setValue(0)
        parm.iflg=1;parm.proj_cur=0;return

    if ui.txt_apex_out_1.currentIndex()<24 or ui.txt_apex_out_1.currentIndex()>28:
        ui.txt_crop_1.setDisabled(True)
    else:
        ui.txt_crop_1.setEnabled(True)

    ui.txt_weight_1.setText(txts[3])
    parm.of_weight.append(float(txts[3]))

    ui.txt_crop_1.setText(txts[4])
    parm.apex_crop.append(txts[4])

    if parm.apex_var[0]==28:
        parm.apex_output.append('ACY')
    elif parm.apex_var[0]>=9 and parm.apex_var[0]<=27:
        parm.apex_output.append('SAD')
    elif parm.apex_var[0]>=29 :
        parm.apex_output.append('LWE')
    else:
        parm.apex_output.append('RCH')

    #line 11
    if proj_array[10].upper() != 'NODATA':
        txts = proj_array[10].split(',') 
        if txts[0].isnumeric() == False or txts[1].isnumeric() == True :
            msgbox.msg("Error", "An error occurred while reading Line 11. \n"+"Outlets tab's 2ND column inputs incorrect.")
            ui.progressBar.setValue(0)
            parm.iflg=1;parm.proj_cur=0;return

        ui.txt_sub_2.setText(txts[0])
        parm.apex_outlets.append(int(txts[0]))
        if txts[1].upper()=="MONTHLY":
            parm.obs_dt.append("MONTHLY")
            ui.txt_dt_2.setCurrentIndex(0)
        elif txts[1].upper()=="DAILY":
            parm.obs_dt.append("DAILY")
            ui.txt_dt_2.setCurrentIndex(1)
        elif txts[1].upper()=="YEARLY":
            parm.obs_dt.append("YEARLY")
            ui.txt_dt_2.setCurrentIndex(2)
        else:
            parm.obs_dt.append("YEARLY")
            ui.txt_dt_2.setCurrentIndex(3)

        if txts[2].upper()=="RCH-FLOW":
            parm.apex_var.append(0)
            ui.txt_apex_out_2.setCurrentIndex(0)
        elif txts[2].upper()=="RCH-SEDIMENT":
            parm.apex_var.append(1)
            ui.txt_apex_out_2.setCurrentIndex(1)
        elif txts[2].upper()=="RCH-TN":
            parm.apex_var.append(2)
            ui.txt_apex_out_2.setCurrentIndex(2)
        elif txts[2].upper()=="RCH-TP":
            parm.apex_var.append(3)
            ui.txt_apex_out_2.setCurrentIndex(3)
        elif txts[2].upper()=="RCH-MINERALN":
            parm.apex_var.append(4)
            ui.txt_apex_out_2.setCurrentIndex(4)
        elif txts[2].upper()=="RCH-ORGANICN":
            parm.apex_var.append(5)
            ui.txt_apex_out_2.setCurrentIndex(5)
        elif txts[2].upper()=="RCH-MINERALP":
            parm.apex_var.append(6)
            ui.txt_apex_out_2.setCurrentIndex(6)
        elif txts[2].upper()=="RCH-ORGANICP":
            parm.apex_var.append(7)
            ui.txt_apex_out_2.setCurrentIndex(7)
        elif txts[2].upper()=="RCH-PEST":
            parm.apex_var.append(8)
            ui.txt_apex_out_2.setCurrentIndex(8)
        elif txts[2].upper()=="SUB-WATERYLD":
            parm.apex_var.append(9)
            ui.txt_apex_out_2.setCurrentIndex(9)
        elif txts[2].upper()=="SUB-RUNOFF":
            parm.apex_var.append(10)
            ui.txt_apex_out_2.setCurrentIndex(10)
        elif txts[2].upper()=="SUB-QDR":
            parm.apex_var.append(11)
            ui.txt_apex_out_2.setCurrentIndex(11)
        elif txts[2].upper()=="SUB-RESQ":
            parm.apex_var.append(12)
            ui.txt_apex_out_2.setCurrentIndex(12)
        elif txts[2].upper()=="SUB-PET":
            parm.apex_var.append(13)
            ui.txt_apex_out_2.setCurrentIndex(13)
        elif txts[2].upper()=="SUB-ET":
            parm.apex_var.append(14)
            ui.txt_apex_out_2.setCurrentIndex(14)
        elif txts[2].upper()=="SUB-SW":
            parm.apex_var.append(15)
            ui.txt_apex_out_2.setCurrentIndex(15)
        elif txts[2].upper()=="SUB-SEDYLD":
            parm.apex_var.append(16)
            ui.txt_apex_out_2.setCurrentIndex(16)
        elif txts[2].upper()=="SUB-QN":
            parm.apex_var.append(17)
            ui.txt_apex_out_2.setCurrentIndex(17)
        elif txts[2].upper()=="SUB-QP":
            parm.apex_var.append(18)
            ui.txt_apex_out_2.setCurrentIndex(18)
        elif txts[2].upper()=="SUB-ORGN":
            parm.apex_var.append(19)
            ui.txt_apex_out_2.setCurrentIndex(19)
        elif txts[2].upper()=="SUB-ORGP":
            parm.apex_var.append(20)
            ui.txt_apex_out_2.setCurrentIndex(20)
        elif txts[2].upper()=="SUB-QDRN":
            parm.apex_var.append(21)
            ui.txt_apex_out_2.setCurrentIndex(21)
        elif txts[2].upper()=="SUB-QDRP":
            parm.apex_var.append(22)
            ui.txt_apex_out_2.setCurrentIndex(22)
        elif txts[2].upper()=="SUB-STMP":
            parm.apex_var.append(23)
            ui.txt_apex_out_2.setCurrentIndex(23)
        elif txts[2].upper()=="CRP-BIOM":
            parm.apex_var.append(24)
            ui.txt_apex_out_2.setCurrentIndex(24)
        elif txts[2].upper()=="CRP-STL":
            parm.apex_var.append(25)
            ui.txt_apex_out_2.setCurrentIndex(25)
        elif txts[2].upper()=="CRP-LAI":
            parm.apex_var.append(26)
            ui.txt_apex_out_2.setCurrentIndex(26)
        elif txts[2].upper()=="CRP-CHT":
            parm.apex_var.append(27)
            ui.txt_apex_out_2.setCurrentIndex(27)
        elif txts[2].upper()=="ACY-YLDG":
            parm.apex_var.append(28)
            ui.txt_apex_out_2.setCurrentIndex(28)
        elif txts[2].upper()=="SED-HFLUX":
            parm.apex_var.append(29)
            ui.txt_apex_out_2.setCurrentIndex(29)
        elif txts[2].upper()=="SED-VFLUX":
            parm.apex_var.append(30)
            ui.txt_apex_out_2.setCurrentIndex(30)
        else:
            msgbox.msg("Error", "An error occurred while reading Line 10. \n"+"Outlets tab's 2nd column inputs incorrect.")
            ui.progressBar.setValue(0)
            parm.iflg=1;parm.proj_cur=0;return

        if ui.txt_apex_out_2.currentIndex()<24 or ui.txt_apex_out_2.currentIndex()>28:
            ui.txt_crop_2.setDisabled(True)
        else:
            ui.txt_crop_2.setEnabled(True)

        ui.txt_weight_2.setText(txts[3])
        parm.of_weight.append(float(txts[3]))

        ui.txt_crop_2.setText(txts[4])
        parm.apex_crop.append(txts[4])

        if parm.apex_var[1]==28:
            parm.apex_output.append('ACY')
        elif parm.apex_var[1]>=9 and parm.apex_var[1]<=27:
            parm.apex_output.append('SAD')
        elif parm.apex_var[1]>=29 :
            parm.apex_output.append('LWE')
        else:
            parm.apex_output.append('RCH')

    #line 12
    if proj_array[11].upper() != 'NODATA':
        txts = proj_array[11].split(',') 
        if txts[0].isnumeric() == False or txts[1].isnumeric() == True :
            msgbox.msg("Error", "An error occurred while reading Line 12. \n"+"Outlets tab's 3rd column inputs incorrect.")
            ui.progressBar.setValue(0)
            parm.iflg=1;parm.proj_cur=0;return

        ui.txt_sub_3.setText(txts[0])
        parm.apex_outlets.append(int(txts[0]))
        if txts[1].upper()=="MONTHLY":
            parm.obs_dt.append("MONTHLY")
            ui.txt_dt_3.setCurrentIndex(0)
        elif txts[1].upper()=="DAILY":
            parm.obs_dt.append("DAILY")
            ui.txt_dt_3.setCurrentIndex(1)
        elif txts[1].upper()=="YEARLY":
            parm.obs_dt.append("YEARLY")
            ui.txt_dt_3.setCurrentIndex(2)
        else:
            parm.obs_dt.append("YEARLY")
            ui.txt_dt_3.setCurrentIndex(3)

        if txts[2].upper()=="RCH-FLOW":
            parm.apex_var.append(0)
            ui.txt_apex_out_3.setCurrentIndex(0)
        elif txts[2].upper()=="RCH-SEDIMENT":
            parm.apex_var.append(1)
            ui.txt_apex_out_3.setCurrentIndex(1)
        elif txts[2].upper()=="RCH-TN":
            parm.apex_var.append(2)
            ui.txt_apex_out_3.setCurrentIndex(2)
        elif txts[2].upper()=="RCH-TP":
            parm.apex_var.append(3)
            ui.txt_apex_out_3.setCurrentIndex(3)
        elif txts[2].upper()=="RCH-MINERALN":
            parm.apex_var.append(4)
            ui.txt_apex_out_3.setCurrentIndex(4)
        elif txts[2].upper()=="RCH-ORGANICN":
            parm.apex_var.append(5)
            ui.txt_apex_out_3.setCurrentIndex(5)
        elif txts[2].upper()=="RCH-MINERALP":
            parm.apex_var.append(6)
            ui.txt_apex_out_3.setCurrentIndex(6)
        elif txts[2].upper()=="RCH-ORGANICP":
            parm.apex_var.append(7)
            ui.txt_apex_out_3.setCurrentIndex(7)
        elif txts[2].upper()=="RCH-PEST":
            parm.apex_var.append(8)
            ui.txt_apex_out_3.setCurrentIndex(8)
        elif txts[2].upper()=="SUB-WATERYLD":
            parm.apex_var.append(9)
            ui.txt_apex_out_3.setCurrentIndex(9)
        elif txts[2].upper()=="SUB-RUNOFF":
            parm.apex_var.append(10)
            ui.txt_apex_out_3.setCurrentIndex(10)
        elif txts[2].upper()=="SUB-QDR":
            parm.apex_var.append(11)
            ui.txt_apex_out_3.setCurrentIndex(11)
        elif txts[2].upper()=="SUB-RESQ":
            parm.apex_var.append(12)
            ui.txt_apex_out_3.setCurrentIndex(12)
        elif txts[2].upper()=="SUB-PET":
            parm.apex_var.append(13)
            ui.txt_apex_out_3.setCurrentIndex(13)
        elif txts[2].upper()=="SUB-ET":
            parm.apex_var.append(14)
            ui.txt_apex_out_3.setCurrentIndex(14)
        elif txts[2].upper()=="SUB-SW":
            parm.apex_var.append(15)
            ui.txt_apex_out_3.setCurrentIndex(15)
        elif txts[2].upper()=="SUB-SEDYLD":
            parm.apex_var.append(16)
            ui.txt_apex_out_3.setCurrentIndex(16)
        elif txts[2].upper()=="SUB-QN":
            parm.apex_var.append(17)
            ui.txt_apex_out_3.setCurrentIndex(17)
        elif txts[2].upper()=="SUB-QP":
            parm.apex_var.append(18)
            ui.txt_apex_out_3.setCurrentIndex(18)
        elif txts[2].upper()=="SUB-ORGN":
            parm.apex_var.append(19)
            ui.txt_apex_out_3.setCurrentIndex(19)
        elif txts[2].upper()=="SUB-ORGP":
            parm.apex_var.append(20)
            ui.txt_apex_out_3.setCurrentIndex(20)
        elif txts[2].upper()=="SUB-QDRN":
            parm.apex_var.append(21)
            ui.txt_apex_out_3.setCurrentIndex(21)
        elif txts[2].upper()=="SUB-QDRP":
            parm.apex_var.append(22)
            ui.txt_apex_out_3.setCurrentIndex(22)
        elif txts[2].upper()=="SUB-STMP":
            parm.apex_var.append(23)
            ui.txt_apex_out_3.setCurrentIndex(23)
        elif txts[2].upper()=="CRP-BIOM":
            parm.apex_var.append(24)
            ui.txt_apex_out_3.setCurrentIndex(24)
        elif txts[2].upper()=="CRP-STL":
            parm.apex_var.append(25)
            ui.txt_apex_out_3.setCurrentIndex(25)
        elif txts[2].upper()=="CRP-LAI":
            parm.apex_var.append(26)
            ui.txt_apex_out_3.setCurrentIndex(26)
        elif txts[2].upper()=="CRP-CHT":
            parm.apex_var.append(27)
            ui.txt_apex_out_3.setCurrentIndex(27)
        elif txts[2].upper()=="ACY-YLDG":
            parm.apex_var.append(28)
            ui.txt_apex_out_3.setCurrentIndex(28)
        elif txts[2].upper()=="SED-HFLUX":
            parm.apex_var.append(29)
            ui.txt_apex_out_3.setCurrentIndex(29)
        elif txts[2].upper()=="SED-VFLUX":
            parm.apex_var.append(30)
            ui.txt_apex_out_3.setCurrentIndex(30)
        else:
            msgbox.msg("Error", "An error occurred while reading Line 10. \n"+"Outlets tab's 3rd column inputs incorrect.")
            ui.progressBar.setValue(0)
            parm.iflg=1;parm.proj_cur=0;return

        if ui.txt_apex_out_3.currentIndex()<24 or ui.txt_apex_out_3.currentIndex()>28:
            ui.txt_crop_3.setDisabled(True)
        else:
            ui.txt_crop_3.setEnabled(True)

        ui.txt_weight_3.setText(txts[3])
        parm.of_weight.append(float(txts[3]))

        ui.txt_crop_3.setText(txts[4])
        parm.apex_crop.append(txts[4])

        if parm.apex_var[2]==28:
            parm.apex_output.append('ACY')
        elif parm.apex_var[2]>=9 and parm.apex_var[2]<=27:
            parm.apex_output.append('SAD')
        elif parm.apex_var[2]>=29 :
            parm.apex_output.append('LWE')
        else:
            parm.apex_output.append('RCH')


    #line 13
    if proj_array[12].upper() != 'NODATA':
        txts = proj_array[12].split(',') 
        if txts[0].isnumeric() == False or txts[1].isnumeric() == True :
            msgbox.msg("Error", "An error occurred while reading Line 13. \n"+"Outlets tab's 4th column inputs incorrect.")
            ui.progressBar.setValue(0)
            parm.iflg=1;parm.proj_cur=0;return

        ui.txt_sub_4.setText(txts[0])
        parm.apex_outlets.append(int(txts[0]))
        if txts[1].upper()=="MONTHLY":
            parm.obs_dt.append("MONTHLY")
            ui.txt_dt_4.setCurrentIndex(0)
        elif txts[1].upper()=="DAILY":
            parm.obs_dt.append("DAILY")
            ui.txt_dt_4.setCurrentIndex(1)
        elif txts[1].upper()=="YEARLY":
            parm.obs_dt.append("YEARLY")
            ui.txt_dt_4.setCurrentIndex(2)
        else:
            parm.obs_dt.append("YEARLY")
            ui.txt_dt_4.setCurrentIndex(3)

        if txts[2].upper()=="RCH-FLOW":
            parm.apex_var.append(0)
            ui.txt_apex_out_4.setCurrentIndex(0)
        elif txts[2].upper()=="RCH-SEDIMENT":
            parm.apex_var.append(1)
            ui.txt_apex_out_4.setCurrentIndex(1)
        elif txts[2].upper()=="RCH-TN":
            parm.apex_var.append(2)
            ui.txt_apex_out_4.setCurrentIndex(2)
        elif txts[2].upper()=="RCH-TP":
            parm.apex_var.append(3)
            ui.txt_apex_out_4.setCurrentIndex(3)
        elif txts[2].upper()=="RCH-MINERALN":
            parm.apex_var.append(4)
            ui.txt_apex_out_4.setCurrentIndex(4)
        elif txts[2].upper()=="RCH-ORGANICN":
            parm.apex_var.append(5)
            ui.txt_apex_out_4.setCurrentIndex(5)
        elif txts[2].upper()=="RCH-MINERALP":
            parm.apex_var.append(6)
            ui.txt_apex_out_4.setCurrentIndex(6)
        elif txts[2].upper()=="RCH-ORGANICP":
            parm.apex_var.append(7)
            ui.txt_apex_out_4.setCurrentIndex(7)
        elif txts[2].upper()=="RCH-PEST":
            parm.apex_var.append(8)
            ui.txt_apex_out_4.setCurrentIndex(8)
        elif txts[2].upper()=="SUB-WATERYLD":
            parm.apex_var.append(9)
            ui.txt_apex_out_4.setCurrentIndex(9)
        elif txts[2].upper()=="SUB-RUNOFF":
            parm.apex_var.append(10)
            ui.txt_apex_out_4.setCurrentIndex(10)
        elif txts[2].upper()=="SUB-QDR":
            parm.apex_var.append(11)
            ui.txt_apex_out_4.setCurrentIndex(11)
        elif txts[2].upper()=="SUB-RESQ":
            parm.apex_var.append(12)
            ui.txt_apex_out_4.setCurrentIndex(12)
        elif txts[2].upper()=="SUB-PET":
            parm.apex_var.append(13)
            ui.txt_apex_out_4.setCurrentIndex(13)
        elif txts[2].upper()=="SUB-ET":
            parm.apex_var.append(14)
            ui.txt_apex_out_4.setCurrentIndex(14)
        elif txts[2].upper()=="SUB-SW":
            parm.apex_var.append(15)
            ui.txt_apex_out_4.setCurrentIndex(15)
        elif txts[2].upper()=="SUB-SEDYLD":
            parm.apex_var.append(16)
            ui.txt_apex_out_4.setCurrentIndex(16)
        elif txts[2].upper()=="SUB-QN":
            parm.apex_var.append(17)
            ui.txt_apex_out_4.setCurrentIndex(17)
        elif txts[2].upper()=="SUB-QP":
            parm.apex_var.append(18)
            ui.txt_apex_out_4.setCurrentIndex(18)
        elif txts[2].upper()=="SUB-ORGN":
            parm.apex_var.append(19)
            ui.txt_apex_out_4.setCurrentIndex(19)
        elif txts[2].upper()=="SUB-ORGP":
            parm.apex_var.append(20)
            ui.txt_apex_out_4.setCurrentIndex(20)
        elif txts[2].upper()=="SUB-QDRN":
            parm.apex_var.append(21)
            ui.txt_apex_out_4.setCurrentIndex(21)
        elif txts[2].upper()=="SUB-QDRP":
            parm.apex_var.append(22)
            ui.txt_apex_out_4.setCurrentIndex(22)
        elif txts[2].upper()=="SUB-STMP":
            parm.apex_var.append(23)
            ui.txt_apex_out_4.setCurrentIndex(23)
        elif txts[2].upper()=="CRP-BIOM":
            parm.apex_var.append(24)
            ui.txt_apex_out_4.setCurrentIndex(24)
        elif txts[2].upper()=="CRP-STL":
            parm.apex_var.append(25)
            ui.txt_apex_out_4.setCurrentIndex(25)
        elif txts[2].upper()=="CRP-LAI":
            parm.apex_var.append(26)
            ui.txt_apex_out_4.setCurrentIndex(26)
        elif txts[2].upper()=="CRP-CHT":
            parm.apex_var.append(27)
            ui.txt_apex_out_4.setCurrentIndex(27)
        elif txts[2].upper()=="ACY-YLDG":
            parm.apex_var.append(28)
            ui.txt_apex_out_4.setCurrentIndex(28)
        elif txts[2].upper()=="SED-HFLUX":
            parm.apex_var.append(29)
            ui.txt_apex_out_4.setCurrentIndex(29)
        elif txts[2].upper()=="SED-VFLUX":
            parm.apex_var.append(30)
            ui.txt_apex_out_4.setCurrentIndex(30)
        else:
            msgbox.msg("Error", "An error occurred while reading Line 10. \n"+"Outlets tab's 4th column inputs incorrect.")
            ui.progressBar.setValue(0)
            parm.iflg=1;parm.proj_cur=0;return

        if ui.txt_apex_out_4.currentIndex()<24 or ui.txt_apex_out_4.currentIndex()>28:
            ui.txt_crop_4.setDisabled(True)
        else:
            ui.txt_crop_4.setEnabled(True)

        ui.txt_weight_4.setText(txts[3])
        parm.of_weight.append(float(txts[3]))

        ui.txt_crop_4.setText(txts[4])
        parm.apex_crop.append(txts[4])

        if parm.apex_var[3]==28:
            parm.apex_output.append('ACY')
        elif parm.apex_var[3]>=9 and parm.apex_var[3]<=27:
            parm.apex_output.append('SAD')
        elif parm.apex_var[3]>=29 :
            parm.apex_output.append('LWE')
        else:
            parm.apex_output.append('RCH')

    ui.progressBar.setValue(50)
    qApp.processEvents()
    time.sleep(0.1)

    #APEX Parameters tab
    for i in range(13,50):
        j = i - 13
        txts = proj_array[i].split(',')   
        ui.tbl_apex_parameters.setItem(j, 0, QTableWidgetItem(txts[0]))
        ui.tbl_apex_parameters.setItem(j, 1, QTableWidgetItem(txts[1]))
        ui.tbl_apex_parameters.setItem(j, 2, QTableWidgetItem(txts[2]))
        ui.tbl_apex_parameters.setItem(j, 3, QTableWidgetItem(txts[3]))
        ui.tbl_apex_parameters.setItem(j, 4, QTableWidgetItem(txts[4]))
        ui.tbl_apex_parameters.setItem(j, 5, QTableWidgetItem(txts[5]))
    
    #PARMs tab
    for i in range(50,160):
        j = i - 50
        txts = proj_array[i].split(',')   
        ui.tbl_parms.setItem(j, 0, QTableWidgetItem(txts[0]))
        ui.tbl_parms.setItem(j, 1, QTableWidgetItem(txts[1]))
        ui.tbl_parms.setItem(j, 2, QTableWidgetItem(txts[2]))
        ui.tbl_parms.setItem(j, 3, QTableWidgetItem(txts[3]))
        ui.tbl_parms.setItem(j, 4, QTableWidgetItem(txts[4]))

    ui.progressBar.setValue(75)
    qApp.processEvents()
    time.sleep(0.1)

    #Crop Parameters tab
    for i in range(160,186):
        j = i - 160
        txts = proj_array[i].split(',')   
        ui.tbl_crop_parameters.setItem(j, 0, QTableWidgetItem(txts[0]))
        ui.tbl_crop_parameters.setItem(j, 1, QTableWidgetItem(txts[1]))
        ui.tbl_crop_parameters.setItem(j, 2, QTableWidgetItem(txts[2]))
        ui.tbl_crop_parameters.setItem(j, 3, QTableWidgetItem(txts[3]))
        ui.tbl_crop_parameters.setItem(j, 4, QTableWidgetItem(txts[4]))
        ui.tbl_crop_parameters.setItem(j, 5, QTableWidgetItem(txts[5]))

    #Constraints tab
    for i in range(186,192):
        j = i - 186
        txts = proj_array[i].split(',')   
        ui.tbl_constraints.setItem(j, 0, QTableWidgetItem(txts[0]))
        ui.tbl_constraints.setItem(j, 1, QTableWidgetItem(txts[1]))
        ui.tbl_constraints.setItem(j, 2, QTableWidgetItem(txts[2]))
        ui.tbl_constraints.setItem(j, 3, QTableWidgetItem(txts[3]))
        ui.tbl_constraints.setItem(j, 4, QTableWidgetItem(txts[4]))
        ui.tbl_constraints.setItem(j, 5, QTableWidgetItem(txts[5]))

    ui.messages.append('A project is opened: ')
    ui.messages.append( parm.file_proj  + '\n')
    
    ui.progressBar.setValue(100)
    qApp.processEvents()
    time.sleep(0.5)
    ui.progressBar.setValue(0)


def saveas_proj():
    if parm.proj_cur==0:
        parm.error_msg = "Create a new project or open a project \n"+" before saving the project."
        msgbox.msg("Error", parm.error_msg)
        ui.messages.append("Error: "+parm.error_msg)
        parm.iflg=1; return

    str = 'Save the CUTE project As...'
    nameproj = QFileDialog.getSaveFileName(None,str,parm.path_proj,"Project File (*.cproj)")
    if nameproj[0]=='':return
    parm.file_proj = nameproj[0]
    parm.path_proj = os.path.dirname(nameproj[0])
    parm.path_proj = os.path.normpath(parm.path_proj)
    parm.file_proj = os.path.normpath(parm.file_proj)

    ui.setWindowTitle(parm.file_proj)
    save_proj()

def save_proj():
    #os.remove(parm.file_proj)
    if parm.proj_cur==0:
        parm.error_msg = "Create a new project or open a project \n"+" before saving the project."
        msgbox.msg("Error", parm.error_msg)
        ui.messages.append("Error: "+parm.error_msg)
        parm.iflg=1; return
    
    #Save the project path
    os.chdir(parm.cute_path)
    if os.path.isfile('path.dat')==True:
        os.remove('path.dat')

    fnam = open('path.dat', 'w')
    fnam.writelines(parm.path_proj)
    fnam.close()

    try:
        fnam = open(parm.file_proj, 'w')
    except IOError:
        return

    #fnam.write(parm.path_proj  + '\n')
    fnam.write(parm.path_apex  + '\n') #line 1

    if ui.rb_calibration.isChecked():
        parm.cute_option='Calib'
    else:
        if ui.rb_sa_sobol.isChecked():
            parm.cute_option='Sobol'
        else:
            parm.cute_option='FAST'

    fnam.write(parm.cute_option  + '\n') #line 2
    fnam.write(ui.txt_num_sa_interval.text()  + '\n') #line3

    strg = ui.txt_calibration_start_date.text() + ',' + ui.txt_calibration_end_date.text()  + '\n'
    fnam.write(strg) #line4
    strg = ui.txt_validation_start_date.text() + ',' + ui.txt_validation_end_date.text()  + '\n'
    fnam.write(strg) #line5

    fnam.write(ui.txt_dds_total_num.text()  + '\n') #line6
    fnam.write(ui.txt_stat.currentText() + '\n') #line7
    fnam.write(ui.drop_calib_option.currentText()  + '\n') #line8
    fnam.write(ui.drop_dds_init_cond.currentText()  + '\n') #line9
    strg =ui.txt_sub_1.text() \
            + ',' + ui.txt_dt_1.currentText() \
            + ',' + ui.txt_apex_out_1.currentText() \
            + ',' + ui.txt_weight_1.text()  \
            + ',' + ui.txt_crop_1.text()  + '\n'
    fnam.write(strg) #line10
    if ui.txt_sub_2.text():
        strg =ui.txt_sub_2.text() \
                + ',' + ui.txt_dt_2.currentText() \
                + ',' + ui.txt_apex_out_2.currentText() \
                + ',' + ui.txt_weight_2.text()  \
                + ',' + ui.txt_crop_2.text()  + '\n'
    else:
        strg = 'NoData'  + '\n'
    fnam.write(strg)
    if ui.txt_sub_3.text():
        strg =ui.txt_sub_3.text() \
                + ',' + ui.txt_dt_3.currentText() \
                + ',' + ui.txt_apex_out_3.currentText() \
                + ',' + ui.txt_weight_3.text()  \
                + ',' + ui.txt_crop_3.text()  + '\n'
    else:
        strg = 'NoData'  + '\n'
    fnam.write(strg)
    if ui.txt_sub_4.text():
        strg =ui.txt_sub_4.text() \
                + ',' + ui.txt_dt_4.currentText() \
                + ',' + ui.txt_apex_out_4.currentText() \
                + ',' + ui.txt_weight_4.text()  \
                + ',' + ui.txt_crop_4.text()  + '\n'
    else:
        strg = 'NoData'  + '\n'
    fnam.write(strg)

    allRows = ui.tbl_apex_parameters.rowCount()
    for row in range(0,allRows):
        strg =ui.tbl_apex_parameters.item(row,0).text()  \
                + ',' + ui.tbl_apex_parameters.item(row,1).text() \
                + ',' + ui.tbl_apex_parameters.item(row,2).text() \
                + ',' + ui.tbl_apex_parameters.item(row,3).text() \
                + ',' + ui.tbl_apex_parameters.item(row,4).text() \
                + ',' + ui.tbl_apex_parameters.item(row,5).text()  + '\n'
        fnam.write(strg)

    allRows = ui.tbl_parms.rowCount()
    for row in range(0,allRows):
        strg =ui.tbl_parms.item(row,0).text()  \
                + ',' + ui.tbl_parms.item(row,1).text() \
                + ',' + ui.tbl_parms.item(row,2).text() \
                + ',' + ui.tbl_parms.item(row,3).text() \
                + ',' + ui.tbl_parms.item(row,4).text()   + '\n'
        fnam.write(strg)

    allRows = ui.tbl_crop_parameters.rowCount()
    for row in range(0,allRows):
        strg =ui.tbl_crop_parameters.item(row,0).text()  \
                + ',' + ui.tbl_crop_parameters.item(row,1).text() \
                + ',' + ui.tbl_crop_parameters.item(row,2).text() \
                + ',' + ui.tbl_crop_parameters.item(row,3).text() \
                + ',' + ui.tbl_crop_parameters.item(row,4).text() \
                + ',' + ui.tbl_crop_parameters.item(row,5).text()   + '\n'
        fnam.write(strg)

    allRows = ui.tbl_constraints.rowCount()
    for row in range(0,allRows):
        strg =ui.tbl_constraints.item(row,0).text()  \
                + ',' + ui.tbl_constraints.item(row,1).text() \
                + ',' + ui.tbl_constraints.item(row,2).text() \
                + ',' + ui.tbl_constraints.item(row,3).text() \
                + ',' + ui.tbl_constraints.item(row,4).text() \
                + ',' + ui.tbl_constraints.item(row,5).text()   + '\n'
        fnam.write(strg)

    ui.messages.append('Current project is saved to: ')
    ui.messages.append(parm.file_proj+ '\n')

    fnam.close()
   
def btn_confirm():
    #Save settings and input 
    ui.progressBar_1.setValue(5);qApp.processEvents();time.sleep(0.1)

    if parm.proj_cur==0:
        parm.error_msg = "Create a new project or open a project \n"+" before proceeding."
        msgbox.msg("Error", parm.error_msg)
        ui.progressBar_1.setValue(0)
        ui.messages.append("Error: "+parm.error_msg)
        parm.iflg=1; return

    #Check if path and option setting are saved
    if parm.flg_path_save == 0:
        parm.error_msg = "Link APEX files \n in the Main tab."
        msgbox.msg("Error", parm.error_msg)
        ui.progressBar_1.setValue(0)
        ui.messages.append("Error: "+parm.error_msg)
        parm.iflg=1; return

    if ui.chk_validation.isChecked() == True:
        dat=QDate.fromString(ui.txt_validation_start_date.text(),'M/d/yyyy')
        parm.start_val = datetime.date(QDate.year(dat),QDate.month(dat),QDate.day(dat))
        dat=QDate.fromString(ui.txt_validation_end_date.text(),'M/d/yyyy')
        parm.end_val = datetime.date(QDate.year(dat),QDate.month(dat),QDate.day(dat))

        parm.flg_validation=1
    else:
        parm.flg_validation=0

    #grey out irrelevant task options
    reset_config_status()

    #parameter interval for sensitivity analylsis
    parm.SA_n = int(ui.txt_num_sa_interval.text())

    #Outlets setting
    cute_setting_save()
    if parm.iflg == 1: 
        ui.progressBar_1.setValue(0);qApp.processEvents()
        return

    #Check if observed data files exist for the selected subareas/reaches
    if ui.rb_calibration.isChecked():
        filelist=[]
        for file in os.listdir(parm.path_obs):
            if file.endswith(".csv"):
                filelist.append(file)

        for i in range(len(parm.apex_outlets)):
            fn = 'rch_'
            if parm.apex_var[i]<=8:
                fn = 'rch_'
            elif parm.apex_var[i]<=23: 
                fn = 'sub_' 
            elif parm.apex_var[i]<=27: 
                fn = 'crp_' 
            elif parm.apex_var[i]==28: 
                filename = 'obs_crop.csv'
            else:
                fn = 'lwe_' 

            if parm.apex_var[i]!=28: 
                if parm.obs_dt[i].upper() == 'DAILY':
                    filename = fn + 'daily' + str(parm.apex_outlets[i]) + '.csv'

                elif parm.obs_dt[i].upper() == 'MONTHLY':
                    filename = fn + 'monthly' + str(parm.apex_outlets[i]) + '.csv'
                elif parm.obs_dt[i].upper() == 'YEARLY':
                    filename = fn + 'yearly' + str(parm.apex_outlets[i]) + '.csv'

            if not filename in filelist:
                parm.error_msg = filename + ' is not found'
                msgbox.msg("Error ", parm.error_msg)
                ui.messages.append(parm.error_msg+' \n')
                ui.progressBar_1.setValue(0)
                parm.iflg=1; return

    ui.progressBar_1.setValue(10);qApp.processEvents();time.sleep(0.1)

    #calibration setting
    if ui.rb_calibration.isChecked():
        check_cal_setting()
    ui.progressBar_1.setValue(20);qApp.processEvents();time.sleep(0.1)

    read_apex_parameters()
    ui.progressBar_1.setValue(30);qApp.processEvents();time.sleep(0.1)

    read_PARMS_sub()
    ui.progressBar_1.setValue(45);qApp.processEvents();time.sleep(0.1)

    read_CROP_sub()
    ui.progressBar_1.setValue(60);qApp.processEvents();time.sleep(0.1)

    read_constraints_sub()
    ui.progressBar_1.setValue(70);qApp.processEvents();time.sleep(0.1)
    
    #Load APEX parameters user selected after combining.
    parm.par_ID=[]
    parm.par_name=[]
    parm.par_initval=[]
    parm.par_bl=[]
    parm.par_bu=[]
    parm.par_cropname=[]
    parm.par_filename=[]
    id = 0
    for i in range(len(parm.apex_par_name)):
        parm.par_ID.append(id)
        parm.par_name.append(parm.apex_par_name[i])
        parm.par_initval.append(parm.apex_par_initval[i])
        parm.par_bl.append(parm.apex_par_bl[i])
        parm.par_bu.append(parm.apex_par_bu[i])
        parm.par_filename.append(parm.apex_par_filename[i])
        parm.par_cropname.append("")
        id = id + 1
    for i in range(len(parm.PARMS_par_name)):
        parm.par_ID.append(id)
        parm.par_name.append(parm.PARMS_par_name[i])
        parm.par_initval.append(parm.PARMS_par_initval[i])
        parm.par_bl.append(parm.PARMS_par_bl[i])
        parm.par_bu.append(parm.PARMS_par_bu[i])
        parm.par_filename.append("PARM")
        parm.par_cropname.append("")
        id = id + 1
    for i in range(len(parm.CROP_par_name)):
        parm.par_ID.append(id)
        parm.par_name.append(parm.CROP_par_name[i])
        parm.par_initval.append(parm.CROP_par_initval[i])
        parm.par_bl.append(parm.CROP_par_bl[i])
        parm.par_bu.append(parm.CROP_par_bu[i])
        parm.par_filename.append("CROP")
        parm.par_cropname.append(parm.CROP_par_cropname[i])
        id = id + 1
    ui.messages.append("Total number of selected parameters: "+str(id)+'\n')
    if id==0:
        msg = "No parameter is selected!"
        msgbox.msg("Error", msg)
        ui.progressBar_1.setValue(0);qApp.processEvents()
        return

    ui.progressBar_1.setValue(80);qApp.processEvents();time.sleep(0.1)

    #Flag APEX files that are to be updated.
    File_toUpdate.update()
    
    #Set current dir to the project folder
    os.chdir(parm.path_proj)

    #Enable the RUN button for executing the program
    ui.btn_run.setEnabled(True)   
    #save the project file
    ui.progressBar_1.setValue(90);qApp.processEvents();time.sleep(0.1)

    save_proj()

    if ui.rb_calibration.isChecked():
        parm.sa_n_iter = int(ui.txt_dds_total_num.text()) 
        strg = "During auto-calibration, APEX will iterate " + str(parm.sa_n_iter) + " times."
    else:
        if ui.rb_sa_sobol.isChecked():
            parm.sa_n_iter = 2 * parm.SA_n * (len(parm.par_name) + 1)                
        elif ui.rb_sa_fast.isChecked():
        
            if parm.SA_n<64: 
                parm.SA_n = 100
                ui.messages.append('Sampling Interval is increased to 100 for the FAST method.')
            parm.sa_n_iter = parm.SA_n * len(parm.par_name)
            ui.txt_num_sa_interval.setText(str(parm.SA_n))
        strg = "During sensitivity analysis, APEX will iterate " + str(parm.sa_n_iter) + " times."
     
    ui.messages.append(strg + '\n')
    qApp.processEvents();time.sleep(0.1)

    ui.messages.append("Ready to Run...")

    ui.progressBar_1.setValue(100);qApp.processEvents();time.sleep(0.5)
    ui.progressBar_1.setValue(0);qApp.processEvents()
 
def btn_run():
    parm.iflg==0
    os.chdir(parm.path_proj)

    ui.fig.clf()
    if ui.rb_calibration.isChecked():
        #show calibration status window in the plotting
        ax1=ui.fig.add_subplot(211)
        ax1.set_title("Calibration Status")
        ax1.set_ylabel("Current OF")
        ax1.set_xlim (0,parm.dds_ndraw)
        ax1.xaxis.set_visible(False)
        ax2 = ui.fig.add_subplot(212)
        ax2.set_xlabel("# iteration")
        ax2.set_ylabel("Best OF")
        ax2.set_xlim (0,parm.dds_ndraw)
        ui.fig.canvas.draw()

        parm.cute_option='calib'
        parm.SA_orCal=0        
        ui.messages.append('\n'+"Calibration begins..."+'\n')
        qApp.processEvents()
        time.sleep(0.1)        
        calib()
    else:
        if ui.rb_sa_sobol.isChecked():
            parm.cute_option='Sobol'
            parm.SA_orCal=1        
            parm.SA_method=1   # used in SA component (1,2); Method of Sobol=1,FAST=2.        initialize()          
            initialize()
            if parm.iflg==1: return                 
            data_pairing.run()
            ui.messages.append('\n'+"Running Sobol Sensetivity Analysis..."+'\n')
            qApp.processEvents()
            time.sleep(0.1)        
            SA_runs()
            if parm.iflg==1: return                 
        elif ui.rb_sa_fast.isChecked():
            parm.cute_option='FAST'
            parm.SA_orCal=1        
            parm.SA_method=2   # used in SA component (1,2); Method of Sobol=1,FAST=2.        initialize()          
            initialize()
            if parm.iflg==1: return                 
            data_pairing.run()
            ui.messages.append('\n'+"Running FAST Sensetivity Analysis..."+'\n')
            qApp.processEvents()
            time.sleep(0.1)        
            SA_runs()
            if parm.iflg==1: return   

def initialize():
    import shutil, par_SA        
    import APEXCONT, APEXFILE, APEXRUN, apexPRINT, APEXSAD,apexLWE
       
    #Allocate some variables
    parm.start_pred = [0 for x in range(len(parm.apex_outlets))]        
    parm.end_pred = [0 for x in range(len(parm.apex_outlets))]        
    parm.pred_date = [0 for x in range(len(parm.apex_outlets))]        
    parm.pred_datea = [0 for x in range(len(parm.apex_outlets))]        
    parm.pred_val = [0 for x in range(len(parm.apex_outlets))]
    parm.pred_vala = [0 for x in range(len(parm.apex_outlets))]
    parm.iflg=0

    #Read files         
    APEXRUN.read() #Get the APEX run name from APEXRUN.DAT
    if parm.iflg==1: return                 
    APEXCONT.read()
    if parm.iflg==1: return                 
    APEXFILE.read()
    if parm.iflg==1: return                 
    apexPRINT.update()                      
    if parm.iflg==1: return  
    
    #Run APEX
    os.chdir(parm.path_TxtWork)
 
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    retcode = subprocess.Popen('apex1501.exe',startupinfo=startupinfo,creationflags=0x08000000)
    retcode.wait()
    os.chdir(parm.path_proj)  

    APEXSAD.select_file()
    if parm.iflg==1: return                 
             
    #write params.txt for SA 
    if parm.SA_orCal==1:
        parm.NumPar_fSA = 0
        par_SA.read()          
        if parm.iflg==1: return                 
         
def calib():
    #master calibration module                        
    import ObsData
    import QuliRun
    import execute_apex 

    parm.SA_orCal = 0 
    initialize() 
    if parm.iflg==1: return                  
        
    #Allocate variables                    
    parm.cur_best_var = [0 for x in range(len(parm.par_name))] 
    parm.cur_test_var = [0 for x in range(len(parm.par_name))]                 
        
    #Load observation data
    ObsData.read() 
    if parm.iflg==1: 
        ui.messages.append("Error: "+parm.error_msg)
        return                 

    #Prepare paired data                       
    data_pairing.run()
    if parm.iflg==1: return                 
   
    #Conduct DDS sampling and auto calibration
    parm.cur_best_OF=1000       
    parm.cur_test_OF=0
    dds_run.execute_dds()
    if parm.iflg==1: 
        ui.messages.append("Error: "+parm.error_msg)
        return                 

    #Update the dataset in TxtWork folder with the best solution (parameter set corresponding to best objective value)
    parm.cur_test_var = parm.cur_best_var[:]
    execute_apex.run()                
    if parm.iflg==1: return                 
        
    QuliRun.save()
    msgbox.msg("Message", "The calibration has completed.")
    ui.progressBar_1.setValue(0)
    qApp.processEvents()
     
def cute_setting_save(): 
    parm.obs_dt=[]
    parm.of_weight=[]
    parm.apex_outlets=[]
    parm.apex_var=[]
    parm.apex_output=[]
    parm.flg_cute_setting_save = 1
    parm.iflg=0

    #column 1
    try:
        val=int(ui.txt_sub_1.text())
        if val==0: 
            parm.error_msg = "Column 1: Enter Subarea ID."
            msgbox.msg("Error ", parm.error_msg)
            ui.messages.append("Error: "+parm.error_msg)
            parm.iflg=1; return
    except ValueError:
            parm.error_msg = "Column 1: Enter Subarea ID."
            msgbox.msg("Error ", parm.error_msg)
            ui.messages.append("Error: "+parm.error_msg)
            parm.iflg=1; return

    parm.apex_outlets.append(int(ui.txt_sub_1.text()))
    parm.obs_dt.append(ui.txt_dt_1.currentText().upper())
    if ui.txt_apex_out_1.currentText()=="RCH-Flow":
        parm.apex_var.append(0)
    elif ui.txt_apex_out_1.currentText()=="RCH-Sediment":
        parm.apex_var.append(1)
    elif ui.txt_apex_out_1.currentText()=="RCH-TN":
        parm.apex_var.append(2)
    elif ui.txt_apex_out_1.currentText()=="RCH-TP":
        parm.apex_var.append(3)
    elif ui.txt_apex_out_1.currentText()=="RCH-MineralN":
        parm.apex_var.append(4)
    elif ui.txt_apex_out_1.currentText()=="RCH-OrganicN":
        parm.apex_var.append(5)
    elif ui.txt_apex_out_1.currentText()=="RCH-MineralP":
        parm.apex_var.append(6)
    elif ui.txt_apex_out_1.currentText()=="RCH-OrganicP":
        parm.apex_var.append(7)
    elif ui.txt_apex_out_1.currentText()=="RCH-PEST":
        parm.apex_var.append(8)
    elif ui.txt_apex_out_1.currentText()=="SUB-WaterYld":
        parm.apex_var.append(9)
    elif ui.txt_apex_out_1.currentText()=="SUB-Runoff":
        parm.apex_var.append(10)
    elif ui.txt_apex_out_1.currentText()=="SUB-QDr":
        parm.apex_var.append(11)
    elif ui.txt_apex_out_1.currentText()=="SUB-ResQ":
        parm.apex_var.append(12)
    elif ui.txt_apex_out_1.currentText()=="SUB-PET":
        parm.apex_var.append(13)
    elif ui.txt_apex_out_1.currentText()=="SUB-ET":
        parm.apex_var.append(14)
    elif ui.txt_apex_out_1.currentText()=="SUB-SW":
        parm.apex_var.append(15)
    elif ui.txt_apex_out_1.currentText()=="SUB-SedYld":
        parm.apex_var.append(16)
    elif ui.txt_apex_out_1.currentText()=="SUB-QN":
        parm.apex_var.append(17)
    elif ui.txt_apex_out_1.currentText()=="SUB-QP":
        parm.apex_var.append(18)
    elif ui.txt_apex_out_1.currentText()=="SUB-OrgN":
        parm.apex_var.append(19)
    elif ui.txt_apex_out_1.currentText()=="SUB-OrgP":
        parm.apex_var.append(20)
    elif ui.txt_apex_out_1.currentText()=="SUB-QDrN":
        parm.apex_var.append(21)
    elif ui.txt_apex_out_1.currentText()=="SUB-QDrP":
        parm.apex_var.append(22)
    elif ui.txt_apex_out_1.currentText()=="SUB-STMP":
        parm.apex_var.append(23)
    elif ui.txt_apex_out_1.currentText()=="CRP-BIOM":
        parm.apex_var.append(24)
    elif ui.txt_apex_out_1.currentText()=="CRP-STL":
        parm.apex_var.append(25)
    elif ui.txt_apex_out_1.currentText()=="CRP-LAI":
        parm.apex_var.append(26)
    elif ui.txt_apex_out_1.currentText()=="CRP-CHT":
        parm.apex_var.append(27)
    elif ui.txt_apex_out_1.currentText()=="ACY-YLDG":
        parm.apex_var.append(28)
    elif ui.txt_apex_out_1.currentText()=="SED-HFLUX":
        parm.apex_var.append(29)
    elif ui.txt_apex_out_1.currentText()=="SED-VFLUX":
        parm.apex_var.append(30)

    if parm.apex_var[0]==28:
        parm.apex_output.append('ACY')
    elif parm.apex_var[0]>=9 and parm.apex_var[0]<=27:
        parm.apex_output.append('SAD')
    elif parm.apex_var[0]>=29:
        parm.apex_output.append('LWE')
    else:
        parm.apex_output.append('RCH')

    if ui.txt_weight_1.text():
        parm.of_weight.append(float(ui.txt_weight_1.text()))
        if parm.of_weight[0] == 0:
            parm.error_msg = "Column 1: Weighting factor can't be zero."
            msgbox.msg("Error ", parm.error_msg)
            ui.messages.append("Error: "+parm.error_msg)
            parm.iflg=1; return
 
    #consistency check 
    if ui.txt_dt_1.currentText()!="Yearly" and ui.txt_apex_out_1.currentText()=="ACY-YLDG (Yearly)":
        parm.error_msg = "Select Yearly timestep to evaluate YLDG."
        msgbox.msg("Error ", parm.error_msg)
        ui.messages.append("Error: "+parm.error_msg)
        parm.iflg=1; return

    if ui.txt_apex_out_1.currentIndex()>25 and len(ui.txt_crop_1.text())!=4:
            parm.error_msg = "Error message", "Input a 4-letter crop name to continue."
            msgbox.msg("Error ", parm.error_msg)
            ui.messages.append("Error: "+parm.error_msg)
            parm.iflg=1; return

    #column 2
    try:
        val=int(ui.txt_sub_2.text())
        if val==0: 
            parm.error_msg = "Check Subarea ID in Column 2"
            msgbox.msg("Error ", parm.error_msg)
            ui.messages.append("Error: "+parm.error_msg)
            parm.iflg=1; return
    except ValueError:
        return

    parm.apex_outlets.append(int(ui.txt_sub_2.text()))
    parm.obs_dt.append(ui.txt_dt_2.currentText().upper())
    if ui.txt_apex_out_2.currentText()=="RCH-Flow":
        parm.apex_var.append(0)
    elif ui.txt_apex_out_2.currentText()=="RCH-Sediment":
        parm.apex_var.append(1)
    elif ui.txt_apex_out_2.currentText()=="RCH-TN":
        parm.apex_var.append(2)
    elif ui.txt_apex_out_2.currentText()=="RCH-TP":
        parm.apex_var.append(3)
    elif ui.txt_apex_out_2.currentText()=="RCH-MineralN":
        parm.apex_var.append(4)
    elif ui.txt_apex_out_2.currentText()=="RCH-OrganicN":
        parm.apex_var.append(5)
    elif ui.txt_apex_out_2.currentText()=="RCH-MineralP":
        parm.apex_var.append(6)
    elif ui.txt_apex_out_2.currentText()=="RCH-OrganicP":
        parm.apex_var.append(7)
    elif ui.txt_apex_out_2.currentText()=="RCH-PEST":
        parm.apex_var.append(8)
    elif ui.txt_apex_out_2.currentText()=="SUB-WaterYld":
        parm.apex_var.append(9)
    elif ui.txt_apex_out_2.currentText()=="SUB-Runoff":
        parm.apex_var.append(10)
    elif ui.txt_apex_out_2.currentText()=="SUB-QDr":
        parm.apex_var.append(11)
    elif ui.txt_apex_out_2.currentText()=="SUB-ResQ":
        parm.apex_var.append(12)
    elif ui.txt_apex_out_2.currentText()=="SUB-PET":
        parm.apex_var.append(13)
    elif ui.txt_apex_out_2.currentText()=="SUB-ET":
        parm.apex_var.append(14)
    elif ui.txt_apex_out_2.currentText()=="SUB-SW":
        parm.apex_var.append(15)
    elif ui.txt_apex_out_2.currentText()=="SUB-SedYld":
        parm.apex_var.append(16)
    elif ui.txt_apex_out_2.currentText()=="SUB-QN":
        parm.apex_var.append(17)
    elif ui.txt_apex_out_2.currentText()=="SUB-QP":
        parm.apex_var.append(18)
    elif ui.txt_apex_out_2.currentText()=="SUB-OrgN":
        parm.apex_var.append(19)
    elif ui.txt_apex_out_2.currentText()=="SUB-OrgP":
        parm.apex_var.append(20)
    elif ui.txt_apex_out_2.currentText()=="SUB-QDrN":
        parm.apex_var.append(21)
    elif ui.txt_apex_out_2.currentText()=="SUB-QDrP":
        parm.apex_var.append(22)
    elif ui.txt_apex_out_2.currentText()=="SUB-STMP":
        parm.apex_var.append(23)
    elif ui.txt_apex_out_2.currentText()=="CRP-BIOM":
        parm.apex_var.append(24)
    elif ui.txt_apex_out_2.currentText()=="CRP-STL":
        parm.apex_var.append(25)
    elif ui.txt_apex_out_2.currentText()=="CRP-LAI":
        parm.apex_var.append(26)
    elif ui.txt_apex_out_2.currentText()=="CRP-CHT":
        parm.apex_var.append(27)
    elif ui.txt_apex_out_2.currentText()=="ACY-YLDG":
        parm.apex_var.append(28)
    elif ui.txt_apex_out_2.currentText()=="SED-HFLUX":
        parm.apex_var.append(29)
    elif ui.txt_apex_out_2.currentText()=="SED-VFLUX":
        parm.apex_var.append(30)

    if parm.apex_var[1]==28:
        parm.apex_output.append('ACY')
    elif parm.apex_var[1]>=9 and parm.apex_var[1]<=27:
        parm.apex_output.append('SAD')
    elif parm.apex_var[1]>=29 :
        parm.apex_output.append('LWE')
    else:
        parm.apex_output.append('RCH')

    if ui.txt_weight_2.text():
        parm.of_weight.append(float(ui.txt_weight_2.text()))
        if parm.of_weight[1] == 0:
            parm.error_msg = "Column 2: Weighting factor can't be zero."
            msgbox.msg("Error ", parm.error_msg)
            ui.messages.append("Error: "+parm.error_msg)
            parm.iflg=1; return
 
    #consistency check 
    if ui.txt_dt_2.currentText()!="Yearly" and ui.txt_apex_out_2.currentText()=="ACY-YLDG (Yearly)":
        parm.error_msg ="Select Yearly timestep to evaluate YLDG."
        msgbox.msg("Error ", parm.error_msg)
        ui.messages.append("Error: "+parm.error_msg)
        parm.iflg=1; return

    if ui.txt_apex_out_2.currentIndex()>25 and len(ui.txt_crop_2.text())!=4:
        parm.error_msg = "Input a 4-letter crop name to continue."
        msgbox.msg("Error ", parm.error_msg)
        ui.messages.append("Error: "+parm.error_msg)
        parm.iflg=1; return

    #column 3
    try:
        val=int(ui.txt_sub_3.text())
        if val==0: 
            parm.error_msg = "Check Subarea ID in Column 3"
            msgbox.msg("Error ", parm.error_msg)
            ui.messages.append("Error: "+parm.error_msg)
            parm.iflg=1; return
    except ValueError:
        return

    parm.apex_outlets.append(int(ui.txt_sub_3.text()))
    parm.obs_dt.append(ui.txt_dt_3.currentText().upper())
    if ui.txt_apex_out_3.currentText()=="RCH-Flow":
        parm.apex_var.append(0)
    elif ui.txt_apex_out_3.currentText()=="RCH-Sediment":
        parm.apex_var.append(1)
    elif ui.txt_apex_out_3.currentText()=="RCH-TN":
        parm.apex_var.append(2)
    elif ui.txt_apex_out_3.currentText()=="RCH-TP":
        parm.apex_var.append(3)
    elif ui.txt_apex_out_3.currentText()=="RCH-MineralN":
        parm.apex_var.append(4)
    elif ui.txt_apex_out_3.currentText()=="RCH-OrganicN":
        parm.apex_var.append(5)
    elif ui.txt_apex_out_3.currentText()=="RCH-MineralP":
        parm.apex_var.append(6)
    elif ui.txt_apex_out_3.currentText()=="RCH-OrganicP":
        parm.apex_var.append(7)
    elif ui.txt_apex_out_3.currentText()=="RCH-PEST":
        parm.apex_var.append(8)
    elif ui.txt_apex_out_3.currentText()=="SUB-WaterYld":
        parm.apex_var.append(9)
    elif ui.txt_apex_out_3.currentText()=="SUB-Runoff":
        parm.apex_var.append(10)
    elif ui.txt_apex_out_3.currentText()=="SUB-QDr":
        parm.apex_var.append(11)
    elif ui.txt_apex_out_3.currentText()=="SUB-ResQ":
        parm.apex_var.append(12)
    elif ui.txt_apex_out_3.currentText()=="SUB-PET":
        parm.apex_var.append(13)
    elif ui.txt_apex_out_3.currentText()=="SUB-ET":
        parm.apex_var.append(14)
    elif ui.txt_apex_out_3.currentText()=="SUB-SW":
        parm.apex_var.append(15)
    elif ui.txt_apex_out_3.currentText()=="SUB-SedYld":
        parm.apex_var.append(16)
    elif ui.txt_apex_out_3.currentText()=="SUB-QN":
        parm.apex_var.append(17)
    elif ui.txt_apex_out_3.currentText()=="SUB-QP":
        parm.apex_var.append(18)
    elif ui.txt_apex_out_3.currentText()=="SUB-OrgN":
        parm.apex_var.append(19)
    elif ui.txt_apex_out_3.currentText()=="SUB-OrgP":
        parm.apex_var.append(20)
    elif ui.txt_apex_out_3.currentText()=="SUB-QDrN":
        parm.apex_var.append(21)
    elif ui.txt_apex_out_3.currentText()=="SUB-QDrP":
        parm.apex_var.append(22)
    elif ui.txt_apex_out_3.currentText()=="SUB-STMP":
        parm.apex_var.append(23)
    elif ui.txt_apex_out_3.currentText()=="CRP-BIOM":
        parm.apex_var.append(24)
    elif ui.txt_apex_out_3.currentText()=="CRP-STL":
        parm.apex_var.append(25)
    elif ui.txt_apex_out_3.currentText()=="CRP-LAI":
        parm.apex_var.append(26)
    elif ui.txt_apex_out_3.currentText()=="CRP-CHT":
        parm.apex_var.append(27)
    elif ui.txt_apex_out_3.currentText()=="ACY-YLDG":
        parm.apex_var.append(28)
    elif ui.txt_apex_out_3.currentText()=="SED-HFLUX":
        parm.apex_var.append(29)
    elif ui.txt_apex_out_3.currentText()=="SED-VFLUX":
        parm.apex_var.append(30)

    if parm.apex_var[2]==28:
        parm.apex_output.append('ACY')
    elif parm.apex_var[2]>=9 and parm.apex_var[2]<=27:
        parm.apex_output.append('SAD')
    elif parm.apex_var[2]>=29 :
        parm.apex_output.append('LWE')
    else:
        parm.apex_output.append('RCH')

    if ui.txt_weight_3.text():
        parm.of_weight.append(float(ui.txt_weight_3.text()))
        if parm.of_weight[2] == 0:
            parm.error_msg = "Column 3: Weighting factor can't be zero."
            msgbox.msg("Error ", parm.error_msg)
            ui.messages.append("Error: "+parm.error_msg)
            parm.iflg=1; return
 
    #consistency check 
    if ui.txt_dt_3.currentText()!="Yearly" and ui.txt_apex_out_3.currentText()=="ACY-YLDG (Yearly)":
        parm.error_msg = "Select Yearly timestep to evaluate YLDG."
        msgbox.msg("Error ", parm.error_msg)
        ui.messages.append("Error: "+parm.error_msg)
        parm.iflg=1; return

    if ui.txt_apex_out_3.currentIndex()>25 and len(ui.txt_crop_3.text())!=4:
        parm.error_msg = "Input a 4-letter crop name to continue."
        msgbox.msg("Error ", parm.error_msg)
        ui.messages.append("Error: "+parm.error_msg)
        parm.iflg=1; return

    #column 4
    try:
        val=int(ui.txt_sub_4.text())
        if val==0: 
            parm.error_msg = "Check Subarea ID in Column 2"
            msgbox.msg("Error ", parm.error_msg)
            ui.messages.append("Error: "+parm.error_msg)
            parm.iflg=1; return
    except ValueError:
        return

    parm.apex_outlets.append(int(ui.txt_sub_4.text()))
    parm.obs_dt.append(ui.txt_dt_4.currentText().upper())
    if ui.txt_apex_out_4.currentText()=="RCH-Flow":
        parm.apex_var.append(0)
    elif ui.txt_apex_out_4.currentText()=="RCH-Sediment":
        parm.apex_var.append(1)
    elif ui.txt_apex_out_4.currentText()=="RCH-TN":
        parm.apex_var.append(2)
    elif ui.txt_apex_out_4.currentText()=="RCH-TP":
        parm.apex_var.append(3)
    elif ui.txt_apex_out_4.currentText()=="RCH-MineralN":
        parm.apex_var.append(4)
    elif ui.txt_apex_out_4.currentText()=="RCH-OrganicN":
        parm.apex_var.append(5)
    elif ui.txt_apex_out_4.currentText()=="RCH-MineralP":
        parm.apex_var.append(6)
    elif ui.txt_apex_out_4.currentText()=="RCH-OrganicP":
        parm.apex_var.append(7)
    elif ui.txt_apex_out_4.currentText()=="RCH-PEST":
        parm.apex_var.append(8)
    elif ui.txt_apex_out_4.currentText()=="SUB-WaterYld":
        parm.apex_var.append(9)
    elif ui.txt_apex_out_4.currentText()=="SUB-Runoff":
        parm.apex_var.append(10)
    elif ui.txt_apex_out_4.currentText()=="SUB-QDr":
        parm.apex_var.append(11)
    elif ui.txt_apex_out_4.currentText()=="SUB-ResQ":
        parm.apex_var.append(12)
    elif ui.txt_apex_out_4.currentText()=="SUB-PET":
        parm.apex_var.append(13)
    elif ui.txt_apex_out_4.currentText()=="SUB-ET":
        parm.apex_var.append(14)
    elif ui.txt_apex_out_4.currentText()=="SUB-SW":
        parm.apex_var.append(15)
    elif ui.txt_apex_out_4.currentText()=="SUB-SedYld":
        parm.apex_var.append(16)
    elif ui.txt_apex_out_4.currentText()=="SUB-QN":
        parm.apex_var.append(17)
    elif ui.txt_apex_out_4.currentText()=="SUB-QP":
        parm.apex_var.append(18)
    elif ui.txt_apex_out_4.currentText()=="SUB-OrgN":
        parm.apex_var.append(19)
    elif ui.txt_apex_out_4.currentText()=="SUB-OrgP":
        parm.apex_var.append(20)
    elif ui.txt_apex_out_4.currentText()=="SUB-QDrN":
        parm.apex_var.append(21)
    elif ui.txt_apex_out_4.currentText()=="SUB-QDrP":
        parm.apex_var.append(22)
    elif ui.txt_apex_out_4.currentText()=="SUB-STMP":
        parm.apex_var.append(23)
    elif ui.txt_apex_out_4.currentText()=="CRP-BIOM":
        parm.apex_var.append(24)
    elif ui.txt_apex_out_4.currentText()=="CRP-STL":
        parm.apex_var.append(25)
    elif ui.txt_apex_out_4.currentText()=="CRP-LAI":
        parm.apex_var.append(26)
    elif ui.txt_apex_out_4.currentText()=="CRP-CHT":
        parm.apex_var.append(27)
    elif ui.txt_apex_out_4.currentText()=="ACY-YLDG":
        parm.apex_var.append(28)
    elif ui.txt_apex_out_4.currentText()=="SED-HFLUX":
        parm.apex_var.append(29)
    elif ui.txt_apex_out_4.currentText()=="SED-VFLUX":
        parm.apex_var.append(30)

    if parm.apex_var[3]==28:
        parm.apex_output.append('ACY')
    elif parm.apex_var[3]>=9 and parm.apex_var[3]<=27:
        parm.apex_output.append('SAD')
    elif parm.apex_var[3]>=29 :
        parm.apex_output.append('LWE')
    else:
        parm.apex_output.append('RCH')

    if ui.txt_weight_4.text():
        parm.of_weight.append(float(ui.txt_weight_4.text()))
        if parm.of_weight[3] == 0:
            parm.error_msg = "Column 1: Weighting factor can't be zero."
            msgbox.msg("Error ", parm.error_msg)
            ui.messages.append("Error: "+parm.error_msg)
            parm.iflg=1; return
 
    #consistency check 
    if ui.txt_dt_4.currentText()!="Yearly" and ui.txt_apex_out_4.currentText()=="ACY-YLDG (Yearly)":
        parm.error_msg = "Select Yearly timestep to evaluate YLDG."
        msgbox.msg("Error ", parm.error_msg)
        ui.messages.append("Error: "+parm.error_msg)
        parm.iflg=1; return

    if ui.txt_apex_out_4.currentIndex()>25 and len(ui.txt_crop_4.text())!=4:
            parm.error_msg = "Input a 4-letter crop name to continue."
            msgbox.msg("Error ", parm.error_msg)
            ui.messages.append("Error: "+parm.error_msg)
            parm.iflg=1; return


    ui.messages.append("The Outlets setting is saved ok." + '\n')
       
def reset_config_status():
    #enable or disable input options based on user selection of calibration or SA
    if parm.proj_cur==0:
        parm.error_msg = "Create a new project or open a project \n"+" before choosing the task option."
        msgbox.msg("Error ", parm.error_msg)
        ui.messages.append("Error: "+parm.error_msg)
        parm.iflg=1; return

    parm.flg_option_save = 1

    if ui.rb_calibration.isChecked():
        ui.group_cal.setEnabled(True)
        ui.group_sa.setEnabled(False)
        ui.messages.append('Calibration is selected.'+ '\n')

    else:
        ui.group_cal.setEnabled(False)
        ui.group_sa.setEnabled(True)
        if ui.rb_sa_sobol.isChecked():
            ui.messages.append('Sensitivity Analysis using the Sobol method.'+ '\n')
        else:
           ui.messages.append('Sensitivity Analysis using the Fast method.'+ '\n')

def check_cal_setting():
    #check CUTE calibration settings
    if parm.proj_cur==0:
        parm.error_msg = "Create a new project or open a project \n"+" before choosing the task option."
        msgbox.msg("Error ", parm.error_msg)
        ui.messages.append("Error: "+parm.error_msg)
        parm.iflg=1; return

    if int(ui.txt_dds_total_num.text())==0:
        parm.error_msg = "The number of iterations is set to zero. \n"+"Please input a positive value."
        msgbox.msg("Error ", parm.error_msg)
        ui.messages.append("Error: "+parm.error_msg)
        parm.iflg=1; return

    #Save Main tab input
    parm.dds_ndraw = int(ui.txt_dds_total_num.text()) #total number of dds iteration
    parm.dds_stat = ui.txt_stat.currentText()
    if ui.drop_dds_init_cond.currentText()=='User default values': #initial parameters for DDS:  or random search
        parm.dds_useinit = 1  #use user input
    else:
        parm.dds_useinit = 0 #use random sampling

    if ui.drop_calib_option.currentText()=='New calibration': #new or continuing calibration?
        parm.dds_restart = 0 
    else:
        parm.dds_restart = 1

    #calibration/validation periods
    dat=QDate.fromString(ui.txt_calibration_start_date.text(),'M/d/yyyy')
    parm.start_cal = datetime.date(QDate.year(dat),QDate.month(dat),QDate.day(dat))
    dat=QDate.fromString(ui.txt_calibration_end_date.text(),'M/d/yyyy')
    parm.end_cal = datetime.date(QDate.year(dat),QDate.month(dat),QDate.day(dat))
    if parm.start_cal.year > parm.end_cal.year:
        parm.error_msg = "Error: The starting date of calibration is later \n"+"than the ending date.\n"
        msgbox.msg("Error ", parm.error_msg)
        ui.messages.append("Error: "+parm.error_msg)
        parm.iflg=1; return

    #check if calibration/validation period matches APEX simulation period
    fnam = parm.path_TxtWork + '\APEXCONT.DAT'
    try:
        parm.apexcont = open(fnam, 'r')
    except:
        #Print error message and exit
        parm.error_msg =  fnam + " is not found. \n"+"while evaluating the calibration period."
        msgbox.msg("Error ", parm.error_msg)
        ui.messages.append("Error: "+parm.error_msg)
        parm.iflg=1; return 
    lnum = 1
    for txtline in parm.apexcont:
        if lnum == 1: 
            dum = ' '.join(txtline.split())
            data = dum.split(' ')
            parm.txt_nbyr = int(float(data[0]))   #NBYR: Number of years simulated.
            parm.txt_iyr = int(float(data[1]))    #IYR: Beginning year of simulation.
            parm.txt_imo = int(float(data[2]))   #IMO: Beginning month of simulation.
            parm.txt_ida = int(float(data[3]))  #IDA: Beginning julian day of simulation.
            lnum = 0

    if parm.start_cal.year>parm.txt_iyr+parm.txt_nbyr-1 or parm.end_cal.year<parm.txt_iyr:
        spd=data[1]+"-"+str(parm.txt_iyr+parm.txt_nbyr-1)
        parm.error_msg = "The calibration period does not match \n"+"the APEX simulation period in APEXCONT. \n"+"APEX Simulation period: \n"+spd
        msgbox.msg("Error ", parm.error_msg)
        ui.messages.append("Error: "+parm.error_msg)
        parm.iflg=1; return

    if ui.chk_validation.isChecked() == True:
        dat=QDate.fromString(ui.txt_validation_start_date.text(),'M/d/yyyy')
        parm.start_val = datetime.date(QDate.year(dat),QDate.month(dat),QDate.day(dat))
        dat=QDate.fromString(ui.txt_validation_end_date.text(),'M/d/yyyy')
        parm.end_val = datetime.date(QDate.year(dat),QDate.month(dat),QDate.day(dat))

        if parm.start_val.year>parm.txt_iyr+parm.txt_nbyr-1 or parm.end_val.year<parm.txt_iyr:
            parm.error_msg = "The validation period does not match \n"+"the APEX simulation period in APEXCONT."
            msgbox.msg("Error ", parm.error_msg)
            ui.messages.append("Error: "+parm.error_msg)
            parm.iflg=1; return

    ui.messages.append("Calibration setting checked ok.")
 
def set_path():
    #
    if parm.proj_cur==0:
        parm.error_msg = "Create a new project or open a project \n"+" before linking the APEX folder."
        msgbox.msg("Error ", parm.error_msg)
        ui.messages.append("Error: "+parm.error_msg)
        parm.iflg=1; return

    parm.flg_path_save = 1
    parm.path_apex = ui.path_apex.text()
    parm.path_TxtInout = parm.path_proj + '\TxtInOut'
    parm.path_TxtWork = parm.path_proj + '\TxtWork'
    parm.path_obs = parm.path_proj + '\Obs'
    parm.iflg=0

    if not os.path.exists(parm.path_apex):
        parm.error_msg = "The APEX folder \n does not exist."
        msgbox.msg("Error ", parm.error_msg)
        ui.messages.append("Error: "+parm.error_msg)
        parm.iflg=1; return

    #Copy APEX dataset into TxtInout 
    if parm.path_apex.lower() != parm.path_TxtInout.lower():
        if os.path.exists(parm.path_TxtInout):
            try:
                shutil.rmtree(parm.path_TxtInout)
            except:
                parm.error_msg = "An error occurred while deleting the old TxtInOut folder. \n Check if any file in TxtInOut is open."
                msgbox.msg("Error ", parm.error_msg)
                ui.progressBar.setValue(0)
                ui.messages.append("Error: "+parm.error_msg)
                parm.iflg=1; return
        ui.progressBar.setValue(25)
        time.sleep(0.1)
        qApp.processEvents()
 
        try:
            shutil.copytree(parm.path_apex,parm.path_TxtInout)
        except:
            #Print error message and exit
            parm.error_msg = "An error occurred while copying APEX files \n from the APEX folder to the project folder.\n Check if any file in these folders is open."
            msgbox.msg("Error ", parm.error_msg)
            ui.progressBar.setValue(0)
            ui.messages.append("Error: "+parm.error_msg)
            parm.iflg=1; return
    ui.progressBar.setValue(50)
    time.sleep(0.1)
    qApp.processEvents()

    #Copy APEX dataset into TxtWork
    if os.path.exists(parm.path_TxtWork):
        try:
            shutil.rmtree(parm.path_TxtWork)
        except:
            #Print error message and exit
            parm.error_msg = "An error occurred while deleting the old TxtWork folder. \n Check if any file in TxtWork is open."
            msgbox.msg("Error ", parm.error_msg)
            ui.progressBar.setValue(0)
            ui.messages.append("Error: "+parm.error_msg)
            parm.iflg=1; return
    
    ui.progressBar.setValue(75)
    time.sleep(0.1)
    qApp.processEvents()

    try:
        shutil.copytree(parm.path_apex,parm.path_TxtWork)
    except:
        #Print error message and exit
        parm.error_msg = "An error occurred while copying \n APEX files from TxtInOut to TxtWork.\n Check if any file in these folders is open."
        msgbox.msg("Error ", parm.error_msg)
        ui.progressBar.setValue(0)
        ui.messages.append("Error: "+parm.error_msg)
        parm.iflg=1; return

    ui.messages.append('APEX input files are copied to ' + parm.path_TxtInout + '\n')
    ui.progressBar.setValue(100)
    qApp.processEvents()
    time.sleep(0.5)
    ui.progressBar.setValue(0)
    qApp.processEvents()

def read_inputparameters():
    #APEX Parameters tab
    read_apex_parameters()
   
    flg = len(parm.apex_par_name)
    msgbox.msg("Message", str(flg) + " parameters are selected.")

def read_apex_parameters():
    parm.flg_apexparameters_save=1
    #read APEX Parameters table 
    parm.apex_par_name=[]
    parm.apex_par_initval=[]
    parm.apex_par_bl=[]
    parm.apex_par_bu=[]
    parm.apex_par_filename=[]

    allRows = ui.tbl_apex_parameters.rowCount()
    for row in range(0,allRows):
        if int(ui.tbl_apex_parameters.item(row,1).text()) != 0:
            parm.apex_par_name.append(ui.tbl_apex_parameters.item(row,0).text())
            parm.apex_par_initval.append(float(ui.tbl_apex_parameters.item(row,2).text()))
            parm.apex_par_bl.append(float(ui.tbl_apex_parameters.item(row,3).text()))
            parm.apex_par_bu.append(float(ui.tbl_apex_parameters.item(row,4).text()))
            parm.apex_par_filename.append(ui.tbl_apex_parameters.item(row,5).text())
    msg1=' '
    ui.messages.append('APEX Parameters selected: '+msg1.join(parm.apex_par_name)+ '\n')

def read_PARMS():
    #PARMs tab
    read_PARMS_sub()
   
    flg = len(parm.PARMS_par_name)
    msgbox.msg("Message", str(flg) + " parameters are selected.")

def read_PARMS_sub():
    parm.flg_parms_save=1
    #read PARMs table 
    parm.PARMS_par_name=[]
    parm.PARMS_par_initval=[]
    parm.PARMS_par_bl=[]
    parm.PARMS_par_bu=[]

    allRows = ui.tbl_parms.rowCount()
    for row in range(0,allRows):
        if int(ui.tbl_parms.item(row,1).text()) != 0:
            parm.PARMS_par_name.append(ui.tbl_parms.item(row,0).text())
            parm.PARMS_par_initval.append(float(ui.tbl_parms.item(row,2).text()))
            parm.PARMS_par_bl.append(float(ui.tbl_parms.item(row,3).text()))
            parm.PARMS_par_bu.append(float(ui.tbl_parms.item(row,4).text()))
    msg1=' '
    ui.messages.append('PARMs selected: '+msg1.join(parm.PARMS_par_name)+ '\n')

def read_CROP():

    read_CROP_sub()
    flg = len(parm.CROP_par_name)
    msgbox.msg("Message", str(flg) + " parameters are selected.")

def read_CROP_sub():
    parm.flg_crop_save=1
    #read Crop Parameters table 
    parm.CROP_par_name=[]
    parm.CROP_par_initval=[]
    parm.CROP_par_bl=[]
    parm.CROP_par_bu=[]
    parm.CROP_par_cropname=[]

    allRows = ui.tbl_crop_parameters.rowCount()
    for row in range(0,allRows):
        if int(ui.tbl_crop_parameters.item(row,1).text()) != 0:
            parm.CROP_par_name.append(ui.tbl_crop_parameters.item(row,0).text())
            parm.CROP_par_initval.append(float(ui.tbl_crop_parameters.item(row,2).text()))
            parm.CROP_par_bl.append(float(ui.tbl_crop_parameters.item(row,3).text()))
            parm.CROP_par_bu.append(float(ui.tbl_crop_parameters.item(row,4).text()))
            parm.CROP_par_cropname.append(ui.tbl_crop_parameters.item(row,5).text())
    msg1=' '
    ui.messages.append('Crop Parameers selected: '+msg1.join(parm.CROP_par_name)+ '\n')

def read_constraints():

    read_constraints_sub()
    flg = len(parm.cs_type)
    msg = str(flg) + " constraints are selected."
    msgbox.msg("Message", msg)

def read_constraints_sub():
    #read Constraints table 
    parm.flg_constraints_save = 1
    parm.cs_type=[]
    parm.cs_name=[]
    parm.cs_bl=[]
    parm.cs_bu=[]

    flg = 0
    allRows = ui.tbl_constraints.rowCount()
    for row in range(0,allRows):
        if int(ui.tbl_constraints.item(row,1).text()) != 0:
            parm.cs_name.append(ui.tbl_constraints.item(row,0).text())
            parm.cs_type.append(int(ui.tbl_constraints.item(row,2).text()))
            parm.cs_bl.append(float(ui.tbl_constraints.item(row,3).text()))
            parm.cs_bu.append(float(ui.tbl_constraints.item(row,4).text()))
            if int(ui.tbl_constraints.item(row,2).text())==100:
                parm.cs100=1
            flg = flg + 1

    if flg==0:
        parm.cs_on = 0
    else:
        parm.cs_on = 1
       
    msg1=' '
    ui.messages.append('Constraints selected: '+msg1.join(parm.cs_name)+ '\n')

class dds_run():
    def execute_dds():
        import execute_apex
        import var_sample
        from random import randint
        import numpy as np
        import time
        import math
        import parm

        ui.fig.clf()
        ax1=ui.fig.add_subplot(211)
        ax1.set_title("Calibration Status")
        ax1.set_ylabel("Current OF")
        ax1.set_xlim (0,parm.dds_ndraw)
        ax1.xaxis.set_visible(False)
        ax2 = ui.fig.add_subplot(212)
        ax2.set_xlabel("# iteration")
        ax2.set_ylabel("Best OF")
        ax2.set_xlim (0,parm.dds_ndraw)

        ui.fig.canvas.draw()

        xval=[]
        yval1=[]
        yval2=[]

        #Open files for writing
        fname = parm.path_proj + '\\dds.out'
        f_ddsout = open(fname, 'w+')
        fname = parm.path_proj + '\\apex.out'
        f_apexout = open(fname, 'w+')
        fname = parm.path_proj + '\\modPerf.out'
        f_modPerfout = open(fname, 'w+')
  
        #Write headers for DDS.out
        txt=str("{:>10}".format('Run#'))
        for j in range(len(parm.cur_test_var)):
            txt = txt + str("{:>10}".format(parm.par_name[j]))         
        txt = txt + str("{:>10}".format('Test_OF')) + str("{:>10}".format('Best_OF')) + '\n'
        f_ddsout.writelines(txt)
    
        #Write headers for modPerf.out (calibration period and validation period)
        txt=str("{:>5}".format('Run#')) + str("{:>8}".format('Outlet')) + str("{:>7}".format('VarID')) 
        txt1=str("{:>10}".format('PBIAS(%)')) + str("{:>10}".format('R2')) + str("{:>10}".format('NS')) 
        txt2=txt1 + str("{:>10}".format('MEAN')) +str("{:>10}".format('STD')) + str("{:>10}".format('RMSE')) + str("{:>10}".format('AD'))
        txt = txt + txt2 + txt2 + '\n'
        f_modPerfout.writelines(txt)

        #Write headers for apex.out
        txt=str("{:>5}".format('Run#')) + str("{:>8}".format('Outlet')) + str("{:>8}".format('VarID')) + str("{:>10}".format('Test_OF')) 
        txt = txt + str("{:>20}".format('Predicted_values-->')) + '\n'
        f_apexout.writelines(txt)

        #DOS window for Message printing
        ui.messages.append("Initiating calibration environments.... \n")
        qApp.processEvents()


        #--------------------------------------------------
        #1. Initialize DDS iterations
        #--------------------------------------------------
    
        init_runs = 5
       # tot_runs = parm.dds_ndraw - init_runs    
        tot_runs = parm.dds_ndraw

        z= tot_runs
        b = np.zeros([1])
        
        #DDS Setting in DDS_SET.dat
        # Case1: Restart = 0 & UseInit = 0   |This is a new calibration and initial parameter values are to be determined by initial random samplings
        # Case2: Restart = 0 & UseInit = 1   |This is a new calibration and initial parameter values are loaded from par_file
        # Case3: Restart = 1                 |This is a continuation from previous run (DDSout.txt) 
        
        if parm.dds_restart==0:
            if parm.dds_useinit==0: #Case1
                for j in range(init_runs):  # for loop from 0 to (init_runs-1), therefore running init_runs times.
                
                    #Screen print #iteration
                    ui.messages.append("   "+str(j+1) + " / "+ str(init_runs))
                    qApp.processEvents()
                    #DDS iteration number
                    parm.dds_icall = j+1
                
                    #Uniform random search for initial runs
                    dds_run.RandomArray()
                    if parm.iflg==1: return                 

                    #Run APEX and compute OF
                    execute_apex.run()
                    if parm.iflg==1: return                 
            
                    #Update the current best results
                    if j == 0 or parm.cur_test_OF < parm.cur_best_OF:
                    #if parm.cur_test_OF < parm.cur_best_OF:
                        parm.cur_best_var = parm.cur_test_var[:]
                        parm.cur_best_OF = parm.cur_test_OF  
        

            elif parm.dds_useinit > 0: #Case2         
             #   tot_runs = parm.dds_ndraw    
                parm.cur_best_var = parm.par_initval
                parm.cur_test_var = parm.cur_best_var[:]            
                parm.dds_icall = 1
            
                #Run APEX and compute OF
                execute_apex.run()
                if parm.iflg==1: return                 

                #Update OF values
                if parm.cur_test_OF<parm.cur_best_OF:
                    parm.cur_best_var = parm.cur_test_var[:]
                    parm.cur_best_OF = parm.cur_test_OF

            #Print output to dds.out, apex.out, and modPerf.out
            dds_run.print_DDSfiles(f_ddsout,f_apexout,f_modPerfout)        

        else:   #Case3 (To do later...)

            #Continue the previous DDS run 
            #loadDDSrun()
            #parm.dds_icall = parm.dds_prerun
            #parm.cur_best_OF = parm.pre_best_OF
            parm.cur_best_OF = parm.cur_best_OF #dummy; delete this line later.

        #--------------------------------------------------
        #2. Implement DDS procedures
        #--------------------------------------------------
        
        #Set current iteration number
        parm.dds_icall += 1
        #ui.messages.append('\n') 
        ui.messages.append("APEX will iterate " + str(parm.dds_ndraw)+ " times. \n") 
        qApp.processEvents()
        ymax = math.ceil(parm.cur_best_OF)


        for i in range(1,parm.dds_ndraw+1):  # remaining runs
            ui.messages.append("   "+str(i) + " / " + str(parm.dds_ndraw) + "  Current:" + "{0:>12.5f}".format(parm.cur_test_OF)+ "  Best:" + "{0:>12.5f}".format(parm.cur_best_OF))
            qApp.processEvents()

            #Probability for APEX variables to be selected for testing in this iteration
            prob_var_select = 1 - math.log(i) / math.log(tot_runs)
            num_var_eval = 0 #Number of parameters be tested in this iteration
            parm.cur_test_var = parm.cur_best_var[:]

            #APEX parameter sampling 
            for j in range(len(parm.par_name)): #iterate for each parameter
                if random.random()<prob_var_select:
                    testvar = parm.cur_test_var[j] 
                    maxval = parm.par_bu[j]
                    minval = parm.par_bl[j]
                
                    #call the perturbation function for parameter sampling
                    var_sample.perturbation(testvar,maxval,minval,parm.dds_pertsize,j)
                    num_var_eval = num_var_eval + 1
        
            #If no parameter is perturbed, still perturb one parameter 
            if num_var_eval==0:
                ri = math.ceil(randint(1,len(parm.par_name))) - 1
                testvar = parm.cur_test_var[ri]
                maxval = parm.par_bu[ri]
                minval = parm.par_bl[ri]
    
                #call the perturbation function for parameter sampling
                var_sample.perturbation(testvar,maxval,minval,parm.dds_pertsize,ri)

            #Run APEX and compute OF
            execute_apex.run()
            if parm.iflg==1: return                 

            #Evaluate constraints
            take_this = 1
            if len(parm.cs_name)>0 and parm.cs_on==1:
                for j in range(len(parm.cs_name)): #iterate for each constraint
                    if parm.cs_type[j]==1:
                        if parm.cs_name[j].lower()=='rto_bf':
                            if parm.baseflow_ratio<parm.cs_bl[j] or parm.baseflow_ratio>parm.cs_bu[j]:
                                take_this = 0
                        elif parm.cs_name[j].lower()=='pet':
                            if parm.pet<parm.cs_bl[j] or parm.pet>parm.cs_bu[j]:
                                take_this = 0
                    elif parm.cs_type[j]==100:
                        if parm.crop_yld[j]<parm.cs_bl[j] or parm.crop_yld[j]>parm.cs_bu[j]:
                            take_this = 0

            #Update the best OF value
            if take_this==1:
                if parm.cur_test_OF<=parm.cur_best_OF:
                    parm.cur_best_var = parm.cur_test_var[:]
                    parm.cur_best_OF = parm.cur_test_OF

            else:
                #Output did not pass constraints guideline
                parm.cur_test_OF = 9999
  

            xval.append(i)
            yval1.append(parm.cur_test_OF)
            yval2.append(parm.cur_best_OF)
    
            # re-plot area 1 of fig
            ax1.clear()
            ax1.set_title("Calibration Status")
            ax1.set_ylabel("Current OF")
            ax1.set_xlim (0,parm.dds_ndraw)
            ax1.xaxis.set_visible(False)
            ax1.plot(xval,yval1,'.-')
            txt='Current OF='+ str("{:6.2f}".format(parm.cur_test_OF))
            xpos=i+0.02*parm.dds_ndraw
            ax1.text(xpos,parm.cur_test_OF,txt)

            # re-plot area 2 of fig 
            ax2.clear()
            ax2.set_xlabel("# iteration")
            ax2.set_ylabel("Best OF")
            ax2.set_xlim (0,parm.dds_ndraw)
            ax2.plot(xval,yval2,'.-')
            txt='Best OF='+ str("{:6.2f}".format(parm.cur_best_OF))
            xpos=i+0.02*parm.dds_ndraw
            ax2.text(xpos,parm.cur_best_OF,txt)

            ui.fig.canvas.draw()
    
            #Print output to dds.out, apex.out, and modPerf.out
            dds_run.print_DDSfiles(f_ddsout,f_apexout,f_modPerfout)

            #progress bar
            ip = int(i/parm.dds_ndraw*100)
            ui.progressBar_1.setValue(ip)
            qApp.processEvents()

            parm.dds_icall += 1
 
        

        f_ddsout.close()
        f_modPerfout.close()
        f_apexout.close()

    def print_DDSfiles(f_ddsout,f_apexout,f_modPerfout):
        dds_run.print_ddsout(f_ddsout)
        dds_run.print_apexout(f_apexout)
        dds_run.print_modPerfout(f_modPerfout)


    def RandomArray():
        nNum = len(parm.par_name)
        e=''
        for j in range(nNum):
            try:
                parm.cur_test_var[j] = float(parm.par_bl[j] + random.random() * (parm.par_bu[j] - parm.par_bl[j]))
            except Exception as e:
                parm.error_msg = str(e)
                msgbox.msg("Error ", parm.error_msg)
                ui.messages.append("Error: "+parm.error_msg)
                parm.iflg=1; return

    def print_ddsout(f_ddsout):
        #Print output to dds.out
        txt = str("{:10d}".format(parm.dds_icall))
        txt = txt + "".join("{:10.3f}".format(n) for n in parm.cur_test_var)
        txt = txt + str("{:10.3f}".format(parm.cur_test_OF)) + str("{:10.3f}".format(parm.cur_best_OF)) + '\n'
        f_ddsout.writelines(txt)
        f_ddsout.flush()

    def print_modPerfout(f_modPerfout):
        import statistics
        #Print model performance statistics to modePerf.out
        if parm.obs_yld_type=='9999':
            if len(parm.pairpCal)<6:
                txt = str("{:5d}".format(parm.dds_icall)) + str("{:5d}".format(9999)) + str("{:10d}".format(parm.apex_var[0])) 
                txt = txt + '%10.3f' % parm.cur_test_OF
            else:
                txt = str("{:5d}".format(parm.dds_icall)) + str("{:5d}".format(9999)) + str("{:10d}".format(parm.apex_var[0])) 
                txt = txt + '%10.3f' % parm.re + '%10.3f' % parm.r2 + '%10.3f' % parm.nse + '%10.3f' % parm.meanpr + '%10.3f' % parm.stdpr
                txt = txt + '%10.3f' % parm.rmse + '%10.3f' % parm.bias      
        else:
            for i in range(len(parm.apex_outlets)):
                txt = str("{:5d}".format(parm.dds_icall)) + str("{:5d}".format(parm.apex_outlets[i])) + str("{:10d}".format(parm.apex_var[i]))
                statistics.performance_indicators(parm.pairpCal[i],parm.pairoCal[i])        
                txt = txt + '%10.3f' % parm.re + '%10.3f' % parm.r2 + '%10.3f' % parm.nse + '%10.3f' % parm.meanpr + '%10.3f' % parm.stdpr
                txt = txt + '%10.3f' % parm.rmse + '%10.3f' % parm.bias        
                if parm.flg_validation==1: 
                    statistics.performance_indicators(parm.pairpVal[i],parm.pairoVal[i])
                    txt = txt + '%10.3f' % parm.re + '%10.3f' % parm.r2 + '%10.3f' % parm.nse + '%10.3f' % parm.meanpr + '%10.3f' % parm.stdpr
                    txt = txt + '%10.3f' % parm.rmse + '%10.3f' % parm.bias

        txt = txt + '\n'
        f_modPerfout.writelines(txt)
        f_modPerfout.flush()

    def print_apexout(f_apexout):
        #Print output to apex.out
        for i in range(len(parm.apex_outlets)):
            txt = str("{:5d}".format(parm.dds_icall)) + str("{:8d}".format(parm.apex_outlets[i])) 
            if parm.obs_date[i]==9999:
                txt = txt + "{:>8}".format(parm.obs_dt[i]) + "{:10.3f}".format(parm.cur_test_OF) + '%10.3f' % parm.pairpCal[i] 
            else:
                txt = txt + str("{:8d}".format(parm.apex_var[i])) + "{:10.3f}".format(parm.cur_test_OF)

                # only paired values for calibration period
                for j in range(len(parm.pairpCal[i])):
                    txt = txt + '%10.3f' % parm.pairpCal[i][j]

                # output for whole simulation peroid
                if parm.start_val != 0: 
                    for j in range(len(parm.pred_vala[i])):  
                        txt = txt + '%10.3f' % parm.pred_vala[i][j]
            txt = txt + '\n'
            f_apexout.writelines(txt)
            f_apexout.flush()
 
def SA_runs():
    #import fast_sampler, fast, sa_util
    import sa_util
    from SALib.sample import saltelli,fast_sampler
    from SALib.analyze import sobol,fast

    params_file = 'params.txt'      
    problem = sa_util.read_param_file(params_file)

    if parm.SA_method == 1:            # Sobol'
        
        param_values = saltelli.sample(problem, parm.SA_n) #2*100(p+1)=1200 (in the case of p=5)
        sa_process(param_values)        
        Y=np.loadtxt('AvgOut_Sobol.csv', delimiter=',',usecols=(0,))
        Si=sobol.analyze(problem,Y, print_to_console = False)     

        fnam = 'Sensitivity_Rank_Sobol.csv'
        try:
            file = open(fnam, 'w')
        except:
            #Print error message and exit
            parm.error_msg = 'Failed to create '+fnam + '.'+'\n'+'Check if the file is already open.'
            msgbox.msg("Error ", parm.error_msg)
            ui.messages.append("Error: "+parm.error_msg)
            parm.iflg=1; return 
        
        file.writelines('Parameter,SI_First,SI_Total'+'\n')        
        si_1st = ["%8.2E" % x for x in Si['S1']]
        si_total = ["%8.2E" % x for x in Si['ST']]

        inum=int(problem['num_vars'])
        for i in range(inum):
            file.writelines(problem['names'][i]+','+si_1st[i]+','+si_total[i]+'\n')
        file.close()

    elif parm.SA_method == 2:            # Fourier Amplitude Sensitivity Analysis (FAST)
        param_values = fast_sampler.sample(problem,parm.SA_n)
        sa_process(param_values)        
        Y=np.loadtxt('AvgOut_FAST.csv', delimiter=',',usecols=(0,))
        Si=fast.analyze(problem,Y, M=4, print_to_console = False)    

        fnam = 'Sensitivity_Rank_FAST.csv'
        try:
            file = open(fnam, 'w')
        except:
            #Print error message and exit
            parm.error_msg = 'Failed to create '+fnam + '.'+'\n'+'Check if the file is already open.'
            msgbox.msg("Error ", parm.error_msg)
            ui.messages.append("Error: "+parm.error_msg)
            parm.iflg=1; return 
        
        file.writelines('Parameter,SI_First,SI_Total'+'\n')        
        si_1st = ["%8.2E" % x for x in Si['S1']]
        si_total = ["%8.2E" % x for x in Si['ST']]

        inum=int(problem['num_vars'])
        for i in range(inum):
            file.writelines(problem['names'][i]+','+si_1st[i]+','+si_total[i]+'\n')
        file.close()
def sa_process(param_values):
    import run_apexForSA
    
    sample_file = parm.cute_option + '_sample_list.txt'   
    np.savetxt(sample_file, param_values, fmt='%.3f',delimiter=' ')
    fsl = open(sample_file, 'r') # open SA sampled file 

    param = []
    z=0
    #print (strg)

    for txtline in fsl:
        irun = "   " + str(z+1) + " / " + str(parm.sa_n_iter)
        #ui.messages.append(irun)
        ui.statusBar.showMessage('Running... '+irun)
        qApp.processEvents();time.sleep(0.1)
        #print (irun)
        txtline = txtline.split(' ')                      
        for i in range(len(parm.par_name)):
            param.append(float(txtline[i]))

        parm.cur_test_var = param[:]
        param = []
        z += 1
        run_apexForSA.run(z)


        ip = z / parm.sa_n_iter * 100
        ui.progressBar_1.setValue(ip);qApp.processEvents();time.sleep(0.1)

    os.chdir(parm.path_proj)  
    fSAyly = 'YlyOut_' + parm.cute_option + '.csv'  # to save selected APEX output (avg and yearly values)                    
    np.savetxt(fSAyly, parm.pred_fSA, fmt='%s',delimiter=',')  
    fSAm = 'AvgOut_' + parm.cute_option + '.csv'  # to save selected average annual APEX output for SA                    
    np.savetxt(fSAm, parm.pred_fSAm, fmt='%s',delimiter=',')  

    ui.messages.append("\n"+"Run completed!!!")
    ui.statusBar.showMessage('')
    ui.progressBar_1.setValue(0);qApp.processEvents()

def save_log_file():
     text=ui.messages.toPlainText()
     fnam = parm.path_proj + '\\log.txt'
     try:
        with open(fnam, 'w') as f:
            f.write(text)
        msgbox.msg('Done',fnam+' \n Saved successfully!')
     except:
        parm.error_msg ='An error occurred \n while writing log.txt'
        msgbox.msg("Error ", parm.error_msg)
        ui.messages.append("Error: "+parm.error_msg)
        parm.iflg=1; return


def sigint_handler(*args):
    """Handler for the SIGINT signal."""
    sys.stderr.write('\r')
    if QMessageBox.question(None, '', "Are you sure you want to quit?",
                            QMessageBox.Yes | QMessageBox.No,
                            QMessageBox.No) == QMessageBox.Yes:
        QApplication.quit()
        
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)  # A new instance of QApplication
    ui = MyWindow()                
    app.exec_()  


    