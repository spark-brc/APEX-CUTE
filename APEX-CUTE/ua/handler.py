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

class uaInit(object):
    def __init__(self, ui):

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
        sel_parm_pars["type"] = "parm"
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
        infos = self.ua_set_info(ui)
        with open(os.path.join(self.main_dir, 'ua_log.log'), "r", encoding="utf8") as f:
            count = 0
            for i, line in enumerate(f.readlines()):
                # font = QFont("monospace")
                # ui.messages.setFont(font)
                if i > 11 and count < 5:
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
                QCoreApplication.processEvents()
                ui.messages.moveCursor(QtGui.QTextCursor.End)

    def ua_set_info(self, ui):
        infos = {
            "Mode":self.mod,
            "Likelihood": ui.comboBox_likelihoods.currentText(),
            "Number of Chains": ui.spinBox_nchains.value(),
            "Number of Runs": ui.lineEdit_Runs.text(),
            "Version": "0.0.0"
        }
        return infos


    # def print_ua_intro(self, ui):
    #     with open(os.path.join(self.main_dir, 'ua_log.log'), "r", encoding="utf8") as f:
    #         for line in f.readlines():
    #             # font = QFont('Source Sans Pro', 10, QFont.Bold)
    #             # font.setLetterSpacing(QFont.PercentageSpacing, 100)
    #             # font.setPixelSize(fontsize)

    #             # ui.messages.insertPlainText(line)
    #             # print(line,  end='')
    #             # time.sleep(0.5)
    #             # QCoreApplication.processEvents()
    #             font = QFont("monospace")
    #             ui.messages.setFont(font)









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
