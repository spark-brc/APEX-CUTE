
import os
import spotpy
import pandas as pd
import numpy as np


class updatePars(object):
    def __init__(self, pars_df):
        self.pars_df = pars_df

    def update_parm_pars(self, updated_pars, parval_len=8):
        parm_pars_df = self.pars_df.loc[self.pars_df['type']=='parm']
        with open("parms.dat", "r") as f:
            content = f.readlines()
        upper_pars = [x.rstrip() for x in content[:35]] 
        core_pars = [x.strip() for x in content[35:46]]
        lower_pars = [x.rstrip() for x in content[46:]]


        n_core_pars = []
        for line in core_pars:
            n_core_pars += [
                str(line[i:i+parval_len]) for i in range(0, len(line), parval_len)
                ]
        parnams = [f"PARM{i}" for i in range(1, len(n_core_pars)+1)]
        core_pars_df = pd.DataFrame({"parnam":parnams, "val":n_core_pars})

        # replace val with new val here ... working
        core_pars_df.loc[core_pars_df["parnam"]=="PARM1"] = f"{new_val:}"
        newdata = core_pars_df.val.values.reshape(11, 10)

        with open("test.dat", 'w') as f:
            for row in newdata:
                f.write("".join(row) + '\n')        


