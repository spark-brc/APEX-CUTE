import parm
import os
import spotpy
import pandas as pd
import numpy as np



class APEX_setup(object):
    def __init__(
        self, ui, parallel="seq", obj_func=None
        ):
        self.proj_dir = parm.path_proj
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
                spotpy.parameter.Uniform(
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

    def load_ua_pars(self):
        pars_df = pd.read_csv(os.path.join(self.ua_dir, "ua_sel_pars.csv"))
        return pars_df

    def parameters(self):
        return spotpy.parameter.generate(self.params)

