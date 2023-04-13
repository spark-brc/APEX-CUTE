import parm
import os
import pandas as pd
import numpy as np
from ua.pars import updatePars
from distutils.dir_util import copy_tree, remove_tree
import subprocess
from ua import parameter
import datetime
from ua.objectivefunctions import rmse
from PyQt5.QtCore import QCoreApplication
import inspect


# FORM_CLASS,_=loadUiType(find_data_file("main.ui"))

class APEX_setup(object):
    def __init__(
        self, ui, parallel="seq", obj_func=None
        ):
        self.obj_func = obj_func
        self.proj_dir = parm.path_proj
        self.curdir = os.getcwd()
        self.ua_dir = os.path.join(self.proj_dir,"UA_Analysis") 
        if ui.radioButton_dream.isChecked():
            self.mod_folder = "DREAM"
        elif ui.radioButton_mcmc.isChecked():
            self.mod_folder = "MCMC"
        elif ui.radioButton_sceua.isChecked():
            self.mod_folder = "SCEUA"
        self.main_dir = os.path.join(self.ua_dir, self.mod_folder)
        os.chdir(self.main_dir)
        self.params = []
        pars_df = self.load_ua_pars()
        for i in range(len(pars_df)):
            self.params.append(
                ua.parameter.Uniform(
                    name=pars_df.iloc[i, 0],
                    low=pars_df.iloc[i, 3],
                    high=pars_df.iloc[i, 4],
                    optguess=np.mean(
                        [float(pars_df.iloc[i, 3]), float(pars_df.iloc[i, 4])]
                    )            
                )
            )
        self.pars_df = pars_df
        self.parallel = parallel
        if self.parallel == "seq":
            pass
        # NOTE & TODO: mpi4py is for linux and ios () 
        if self.parallel == "mpi":
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            self.mpi_size = comm.Get_size()
            self.mpi_rank = comm.Get_rank()

        # NOTE: read all ui setting here then use as initiates
        if ui.rb_user_obs_day.isChecked():
            self.time_step = "day"
        if ui.rb_user_obs_mon.isChecked():
            self.time_step = "month"
        self.rchnum = ui.txt_sub_1.text()
        if ui.rb_user_obs_type_rch.isChecked():
            self.obs_type = "rch"
        if ui.txt_apex_out_1.currentText().upper()=="RCH-FLOW":
            self.obs_nam = "Flow(m3/s)"
        self.obs_path = ui.txt_user_obs_save_path.toPlainText()
        # self.ui = ui
        # APEXCUTE_path_dict = self.dirs_and_paths()
        # os.chdir(APEXCUTE_path_dict['apexcute'])
        # print(inspect.getmodule(ui).__dir__)



        
    # def load_sim(self, wd):
    #     stdate_, eddate_, ptcode = self.get_start_end_step()
    #     if ptcode == 6 and self.time_step == "day":
    #         sim_df = read_output.extract_day_stf()
    #     if ptcode == 6 and self.time_step == "month":
    #         sim_df = read_output.extract_day_stf(wd)
    #         print('nope!')
    #         sim_df = sim_df.resample('M').mean()
    #         sim_df['year'] = sim_df.index.year
    #         sim_df['month'] = sim_df.index.month
    #         sim_df['time'] = sim_df['year'].astype(str) + "-" + sim_df['month'].astype(str)
    #     if ptcode == 3 and self.time_step == "month":
    #         sim_df = read_output.extract_mon_stf(wd)
    #     print(sim_df)
    #     return sim_df
    


    def load_ua_pars(self):
        pars_df = pd.read_csv(os.path.join(self.ua_dir, "ua_sel_pars.csv"))
        return pars_df

    def parameters(self):
        return parameter.generate(self.params)
    
    def update_apex_pars(self, parameters):
        print(f"this iteration's parameters:")
        print(parameters)
        apex_pars_df = self.pars_df   
        apex_pars_df['val'] = parameters
        self.update_parm_pars(apex_pars_df)


    def update_parm_pars(self, updated_pars, parval_len=8):
        """update parm pars

        Args:
            updated_pars (dataframe):  
            parval_len (int, optional): _description_. Defaults to 8.
        """
        new_pars_df = updated_pars.loc[updated_pars['type']=='parm']
        with open("parms.dat", "r") as f:
            content = f.readlines()
        upper_pars = [x.rstrip() for x in content[:35]] 
        core_pars = [x.rstrip() for x in content[35:46]]
        lower_pars = [x.rstrip() for x in content[46:]]
        n_core_pars = []
        for line in core_pars:
            n_core_pars += [
                str(line[i:i+parval_len]) for i in range(0, len(line), parval_len)
                ]
        parnams = [f"PARM{i}" for i in range(1, len(n_core_pars)+1)]
        core_pars_df = pd.DataFrame({"parnam":parnams, "val":n_core_pars})
        for pnam in core_pars_df["parnam"]:
            if pnam in new_pars_df.loc[:, "name"].tolist():
                new_val = "{:8.4f}".format(float(new_pars_df.loc[new_pars_df["name"] == pnam, "val"].tolist()[0]))
                core_pars_df.loc[
                    core_pars_df["parnam"]==pnam, "val"
                    ] = "{:>8}".format(new_val)
        newdata = core_pars_df.loc[:, "val"].values.reshape(11, 10)

        with open("parms.dat", 'w') as f:
            for urow in upper_pars:
                f.write(urow + '\n')
            for row in newdata:
                f.write("".join(row) + '\n')
            for lrow in lower_pars:
                f.write(lrow + '\n')

    # Simulation function must not return values besides for which evaluation values/observed data are available
    def simulation(self, parameters):     
        if self.parallel == "seq":
            call = ""
        elif self.parallel == "mpi":
            # Running n parallel, care has to be taken when files are read or written
            # Therefor we check the ID of the current computer core
            call = str(int(os.environ["OMPI_COMM_WORLD_RANK"]) + 2)
            # And generate a new folder with all underlying files
            copy_tree(self.main_dir, self.main_dir + call)

        elif self.parallel == "mpc":
            # Running n parallel, care has to be taken when files are read or written
            # Therefor we check the ID of the current computer core
            call = str(os.getpid())
            # And generate a new folder with all underlying files
            # os.chdir(self.wd)
            copy_tree(self.main_dir, self.main_dir + call)
            
        else:
            raise "No call variable was assigned"
        self.main_dir_call =self.main_dir + call
        os.chdir(self.main_dir_call)
        try:
            self.update_apex_pars(parameters)

            comline = "APEX1501.exe"
            run_model = subprocess.Popen(comline, cwd=".", stdout=subprocess.DEVNULL)
            # run_model = subprocess.Popen(comline, cwd=".")
            run_model.wait()
            # all_df = self.all_sim_obd(self.main_dir_call) # read all sim_obd from all 
            all_df = self.com_sim_obd()
        except Exception as e:
            raise Exception("Model has failed")
        
        os.chdir(self.curdir)
        # os.chdir(self.main_dir)
        # os.chdir("d:/Projects/Tools/DayCent-CUTE/tools")
        if self.parallel == "mpi" or self.parallel == "mpc":
            remove_tree(self.main_dir + call)
        return all_df['str_sim'].tolist()

    def extract_day_stf(self):
        rch_file = 'SITE2' + '.RCH'
        channels = [2]
        # start_day = "{}/{}/{}".format(parm.txt_ida, parm.txt_imo, parm.txt_iyr)
        # FIXME: import APEXCON.read() causes error
        start_day = "{}/{}/{}".format(1, 1, 1998)
        cali_start_day = "1/1/2000"
        cali_end_day = "12/31/2007"
        for i in channels:
            sim_stf = pd.read_csv(
                            rch_file,
                            delim_whitespace=True,
                            skiprows=9,
                            usecols=[0, 1, 8],
                            names=["idx", "sub", "str_sim"],
                            index_col=0)
            sim_stf = sim_stf.loc["REACH"]
            sim_stf_f = sim_stf.loc[sim_stf["sub"] == int(i)]
            sim_stf_f = sim_stf_f.drop(['sub'], axis=1)
            sim_stf_f.index = pd.date_range(start_day, periods=len(sim_stf_f.str_sim))
            sim_stf_f = sim_stf_f[cali_start_day:cali_end_day]
        return sim_stf_f

    def load_obs(self):
        stdate_, eddate_, ptcode = self.get_start_end_step()
        if self.time_step == "month":
            timestep = "monthly"
        obs_file = "{}_{}{}.csv".format(self.obs_type, timestep, self.rchnum)
        obs_df = pd.read_csv(os.path.join(self.obs_path, obs_file))
        obs_df = obs_df.replace(-999, np.nan)
        obs_df['time'] =obs_df['Year'].astype(str) + "-" + obs_df['Month'].astype(str)
        obs_df = obs_df.loc[:, [self.obs_nam, 'time']]
        return obs_df


    def com_sim_obd(self):
        sim_df = self.extract_day_stf()
        sim_df = sim_df.resample('M').mean()
        sim_df['year'] = sim_df.index.year
        sim_df['month'] = sim_df.index.month
        sim_df['time'] = sim_df['year'].astype(str) + "-" + sim_df['month'].astype(str)
        obs_df = self.load_obs()
        com_so_df = sim_df.merge(obs_df, how='inner', on='time')
        return com_so_df



    def evaluation(self):
        # os.chdir(self.main_dir_call)
        all_df = self.com_sim_obd()
        return all_df[self.obs_nam].tolist()


    def get_start_end_step(self):
        if os.path.isfile(os.path.join(self.main_dir, "APEXCONT.DAT")):
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

    def objectivefunction(self, simulation, evaluation):
        if not self.obj_func:
            like = rmse(evaluation, simulation)
        else:
            like = self.obj_func(evaluation, simulation)
        print("simulation")
        print(len(simulation))
        print("evaluation")
        print(len(evaluation))

        # objectivefunction = spotpy.objectivefunctions.abs_pbias(
        #     evaluation, simulation
        # )
        return like






# if __name__ == "__main__":
#     APEX_setup.update_apex_pars()

