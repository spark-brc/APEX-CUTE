import parm
import os
import pandas as pd
import shutil
from ua.likelihoods import (
                            gaussianLikelihoodMeasErrorOut, gaussianLikelihoodHomoHeteroDataError, LikelihoodAR1WithC,
                            LikelihoodAR1NoC, generalizedLikelihoodFunction, LaplacianLikelihood, SkewedStudentLikelihoodHomoscedastic,
                            SkewedStudentLikelihoodHeteroscedastic, SkewedStudentLikelihoodHeteroscedasticAdvancedARModel,
                            NoisyABCGaussianLikelihood, ABCBoxcarLikelihood, LimitsOfAcceptability, InverseErrorVarianceShapingFactor,
                            NashSutcliffeEfficiencyShapingFactor, ExponentialTransformErrVarShapingFactor, sumOfAbsoluteErrorResiduals

)
import time
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QCoreApplication
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QFont
import subprocess
from ua import read_output, analyzer
import datetime
import csv


us_log_path = os.path.dirname(os.path.abspath( __file__ ))

class uaInit(object):
    def __init__(self, ui):
        self.curdir = os.getcwd()
        self.proj_dir = parm.path_proj
        self.TxtInout = parm.path_TxtInout
        os.chdir(self.proj_dir)
        self.main_dir = os.path.join(self.proj_dir,"UA_Analysis") # create main UA workding directory
        if not os.path.exists(self.main_dir):
            os.makedirs(self.main_dir)
            ui.messages.append(f"'UA_Analysis folder' was created in {self.proj_dir}.")
        if ui.radioButton_dream.isChecked():
            self.mod = "DREAM"
        elif ui.radioButton_mcmc.isChecked():
            self.mod = "MCMC"
        elif ui.radioButton_sceua.isChecked():
            self.mod = "SCEUA"

        # NOTE: read all ui setting here then use as initiates
        self.cali_stdate = ui.txt_calibration_start_date.text()
        self.cali_eddate = ui.txt_calibration_end_date.text()
        if ui.rb_user_obs_day.isChecked():
            self.time_step = "day"
        if ui.rb_user_obs_mon.isChecked():
            self.time_step = "month"
        self.rchnum01 = ui.txt_sub_1.text()
        if ui.rb_user_obs_type_rch.isChecked():
            self.obs_type = "rch"
        if ui.txt_apex_out_1.currentText().upper()=="RCH-FLOW":
            self.obs_nam = "Flow(m3/s)"
        self.obs_path = ui.txt_user_obs_save_path.toPlainText()
        self.obs_num = ui.txt_user_obs_subnum.text()


    def copy_obs_file_uadir(self):
        if self.time_step == "month":
            ts = "monthly"
        obs_file_nam = f"{self.obs_type}_{ts}{self.obs_num}.csv"
        shutil.copy2(os.path.join(self.obs_path, obs_file_nam), os.path.join(self.main_dir))


    def ua_worktree_setup(self):
        mod = self.mod
        if os.path.exists(os.path.join(self.main_dir, mod)):
            qBox = QMessageBox()
            reply = qBox.question(
                                qBox, 'Overwrite?',
                                f"Do you want to overwrite '{mod}' folder?",
                                qBox.Yes, qBox.No)
            if reply == qBox.Yes:
                try:
                    shutil.rmtree(os.path.join(self.main_dir, mod), onerror=self._remove_readonly)#, onerror=del_rw)
                except Exception as e:
                    raise Exception("unable to remove existing worker dir:" + \
                                    "{0}\n{1}".format(os.path.join(self.main_dir, mod),str(e)))
                try:
                    shutil.copytree(self.TxtInout, os.path.join(self.main_dir, mod))
                except Exception as e:
                    raise Exception("unable to copy files from model dir: " + \
                                    "{0} to new main dir: {1}\n{2}".format(self.TxtInout, os.path.join(self.main_dir, mod),str(e)))
        else:
            try:
                shutil.copytree(self.TxtInout, os.path.join(self.main_dir, mod))
            except Exception as e:
                raise Exception("unable to copy files from model dir: " + \
                                "{0} to new main dir: {1}\n{2}".format(self.TxtInout, os.path.join(self.main_dir, mod),str(e)))        
    
    def export_parm_pars(self, ui):
        allRows = ui.tbl_parms.rowCount()
        parm_pars = pd.DataFrame(
            {
                'name': [ui.tbl_parms.item(i, 0).text() for i in range(allRows)],
                'select': [ui.tbl_parms.item(i, 1).text() for i in range(allRows)],
                'default': [ui.tbl_parms.item(i, 2).text() for i in range(allRows)],
                'min': [ui.tbl_parms.item(i, 3).text() for i in range(allRows)],
                'max': [ui.tbl_parms.item(i, 4).text() for i in range(allRows)],
                }
                )
        sel_parm_pars = parm_pars.loc[parm_pars['select'] == str(1)]
        sel_parm_pars.loc[:, "type"] = "parm"
        return sel_parm_pars

    def export_other_pars(self, ui):
        allRows = ui.tbl_apex_parameters.rowCount()
        other_pars = pd.DataFrame(
            {
                'name': [ui.tbl_apex_parameters.item(i, 0).text() for i in range(allRows)],
                'select': [ui.tbl_apex_parameters.item(i, 1).text() for i in range(allRows)],
                'default': [ui.tbl_apex_parameters.item(i, 2).text() for i in range(allRows)],
                'min': [ui.tbl_apex_parameters.item(i, 3).text() for i in range(allRows)],
                'max': [ui.tbl_apex_parameters.item(i, 4).text() for i in range(allRows)],
                'type': [ui.tbl_apex_parameters.item(i, 5).text() for i in range(allRows)],
                }
                )
        sel_other_pars = other_pars.loc[other_pars['select'] == str(1)]
        return sel_other_pars

    def export_crop_pars(self, ui):
        allRows = ui.tbl_crop_parameters.rowCount()
        crop_pars = pd.DataFrame(
            {
                'name': [ui.tbl_crop_parameters.item(i, 0).text() for i in range(allRows)],
                'select': [ui.tbl_crop_parameters.item(i, 1).text() for i in range(allRows)],
                'default': [ui.tbl_crop_parameters.item(i, 2).text() for i in range(allRows)],
                'min': [ui.tbl_crop_parameters.item(i, 3).text() for i in range(allRows)],
                'max': [ui.tbl_crop_parameters.item(i, 4).text() for i in range(allRows)],
                'type': [ui.tbl_crop_parameters.item(i, 5).text() for i in range(allRows)],
                }
                )
        sel_crop_pars = crop_pars.loc[crop_pars['select'] == str(1)]
        return sel_crop_pars
    
    def all_pars(self, ui):
        parm_pars = self.export_parm_pars(ui)
        other_pars = self.export_other_pars(ui)
        crop_pars = self.export_crop_pars(ui)
        all_pars_df = pd.concat([parm_pars, other_pars, crop_pars], axis=0)
        os.chdir(self.main_dir)
        all_pars_df.to_csv('ua_sel_pars.csv', index=False)

    def _remove_readonly(self, func, path, excinfo):
        """remove readonly dirs, apparently only a windows issue
        add to all rmtree calls: shutil.rmtree(**,onerror=remove_readonly), wk"""
        os.chmod(path, 128)  # stat.S_IWRITE==128==normal
        func(path)

    def print_ua_intro(self, ui):
        font = QFont("Consolas")
        ui.messages.setFont(font)
        ui.messages.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        infos = self.ua_set_info(ui)
        with open(os.path.join(us_log_path, 'ua_log.log'), "r", encoding="utf8") as f:
            count = 0
            for i, line in enumerate(f.readlines()):
                # font = QFont("monospace")
                # ui.messages.setFont(font)
                if i > 10 and count < 12:
                    ui.messages.insertPlainText(
                        line.replace('\n', ' ') + 
                        f" | {list(infos.keys())[count]}:" + 
                        f" {list(infos.values())[count]}"
                        "\n")
                    count+=1
                else:
                    ui.messages.insertPlainText(line)
                # ui.messages.moveCursor(QtGui.QTextCursor.End)
                print(line,  end='')
                time.sleep(0.2)
                ui.messages.moveCursor(QtGui.QTextCursor.End)
                QCoreApplication.processEvents()

    def ua_set_info(self, ui):
        infos = {
            "WD": self.main_dir,
            "Mode":self.mod,
            "Likelihood": ui.comboBox_likelihoods.currentText(),
            "NumberChains": ui.spinBox_nchains.value(),
            "NumberRuns": ui.lineEdit_Runs.text(),
            "CalibrationStartDate": self.cali_stdate,
            "CalibrationEndDate": self.cali_eddate,
            "ObservationFile": self.obs_type,
            "ObservationType": self.obs_nam,
            "ReachIDs": self.obs_num,
            "TimeStepObs": self.time_step,
            "Version": "0.0.0"
        }
        return infos


    def initial_run(self, ui):
        qBox = QMessageBox()
        reply = qBox.question(
                            qBox, 'Please, wait until it is done ...',
                            f"We are going to check on initial test run.\n Click Yes to proceed.",
                            qBox.Yes, qBox.No)
        if reply == qBox.Yes:
            ui.messages.append('Initial run start...')
            QCoreApplication.processEvents()
            mod = self.mod
            os.chdir(os.path.join(os.path.join(self.main_dir, mod)))
            comline = "APEX1501.exe"
            run_model = subprocess.Popen(comline, cwd=".")
            # run_model = subprocess.Popen(comline, cwd=".")
            run_model.wait()
            ui.messages.append('...complete!')
            QCoreApplication.processEvents()

    def create_ua_config(self, ui):
        with open(os.path.join(self.main_dir, 'ua_conf.cfg'),'w', newline="") as f:
            w = csv.writer(f)
            w.writerows(self.ua_set_info(ui).items())
        ui.messages.append("'ua_conf.csv' file was created ..." ) 
        QCoreApplication.processEvents()
    # def check_obs_file(self, ui):

    def read_ua_conf(self):
        mod = self.mod
        self.uaconf_df = pd.read_csv(
            os.path.join(self.main_dir, "ua_conf.cfg"),
            names=["idx", "val"],
            index_col=0, header=None)        
        return self.uaconf_df

    def create_ua_sim_obd(self, ui):
        os.chdir(os.path.join(self.main_dir, self.mod))
        df = read_output.create_ua_sim_obd(self.read_ua_conf())
        analyzer.plot_one_one(df)

        os.chdir(self.main_dir)


    # def create_ua_sim_obd(self, ui):
    #     uaconf_df = self.read_ua_conf()
    #     # read rch
    #     sim_df = read_output.extract_day_stf(
    #         uaconf_df.loc["ReachIDs", "val"],
    #         uaconf_df.loc["CalibrationStartDate", "val"],
    #         uaconf_df.loc["CalibrationEndDate", "val"],
    #     )
    #     print(sim_df)




    def get_start_end_step(self):
        mod = self.mod 
        if os.path.isfile(os.path.join(self.main_dir, mod, "APEXCONT.DAT")):
            with open(os.path.join(self.main_dir, 'APEXCONT.DAT'), "r") as f:
                data = [x.strip().split() for x in f if x.strip()]
            numyr = int(data[0][0])
            styr = int(data[0][1])
            stmon = int(data[0][2])
            stday = int(data[0][3])
            ptcode = int(data[0][4])
            edyr = styr + numyr -1
            stdate = datetime.datetime(styr, stmon, 1) + datetime.timedelta(stday - 1)
            eddate = datetime.datetime(edyr, 12, 31)
            stdate_ = stdate.strftime("%m/%d/%Y")
            eddate_ = eddate.strftime("%m/%d/%Y")
            return stdate_, eddate_, ptcode
    
    def read_apexrun(self):
        mod = self.mod
        with open(os.path.join(self.main_dir, mod, "APEXRUN.DAT", "r")) as f:
            data = [x.strip().split() for x in f]
        return data[0][0]



    def likelihoods_list(self):
        llist = {
            "Gaussian likelihood: MEO": gaussianLikelihoodMeasErrorOut,
            "Gaussian likelihood: HHDE":gaussianLikelihoodHomoHeteroDataError, 
            "LikelihoodAR1WithC": LikelihoodAR1WithC
        }

def obj_list(ui):
    dream_list = ["Gaussian likelihood: MEO", "Gaussian likelihood: HHDE", "LikelihoodAR1WithC"]
    mcmc_list = ["idiot"]
    sceua_list = ["good"]
    if ui.radioButton_dream.isChecked():
        ui.comboBox_likelihoods.clear()
        ui.comboBox_likelihoods.addItems(dream_list)
    elif ui.radioButton_mcmc.isChecked():
        ui.comboBox_likelihoods.clear()
        ui.comboBox_likelihoods.addItems(mcmc_list)
    elif ui.radioButton_sceua.isChecked():
        ui.comboBox_likelihoods.clear()
        ui.comboBox_likelihoods.addItems(sceua_list)



def likelihoods_list():
    llist = {
        "Gaussian likelihood: MEO": gaussianLikelihoodMeasErrorOut,
        "Gaussian likelihood: HHDE":gaussianLikelihoodHomoHeteroDataError, 
        "LikelihoodAR1WithC": LikelihoodAR1WithC
        }
    
def dream_activate():
    print('works?')
