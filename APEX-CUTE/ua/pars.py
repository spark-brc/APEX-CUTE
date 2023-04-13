
import os
# import spotpy
import pandas as pd
import numpy as np


class updatePars(object):
    def __init__(self) -> None:
        pass

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

        # replace val with new val here ... working
        
        for pnam in core_pars_df["parnam"]:
            if pnam in new_pars_df.loc[:, "name"].tolist():
                core_pars_df.loc[
                    core_pars_df["parnam"]==pnam, "val"
                    ] = "{:>8}".format(new_pars_df.loc[new_pars_df["name"] == pnam, "val"].tolist()[0])
        newdata = core_pars_df.val.values.reshape(11, 10)

        with open("parms.dat", 'w') as f:
            for urow in upper_pars:
                f.write(urow + '\n')
            for row in newdata:
                f.write("".join(row) + '\n')
            for lrow in lower_pars:
                f.write(lrow + '\n')     


