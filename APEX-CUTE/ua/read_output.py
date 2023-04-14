import parm
import pandas as pd
import numpy as np
import os
import datetime

def create_ua_sim_obd(uadf):
    stdate_, eddate_, sim_ts = get_start_end_step()
    # 3 monthly, 6 daily
    obs_ts = uadf.loc["TimeStepObs","val"]
    obs_file = uadf.loc["ObservationFile","val"]
    obs_type = uadf.loc["ObservationType","val"]
    rch_num = uadf.loc["ReachIDs", "val"]
    cali_stdate = uadf.loc["CalibrationStartDate", "val"]
    cali_eddate = uadf.loc["CalibrationEndDate", "val"]
    if sim_ts == 6 and obs_ts == "month": # extract daily sim and resample monthly
        sim_df = extract_day_stf(
            rch_num, cali_stdate, cali_eddate
            ).resample('M').mean()
        sim_df['year'] = sim_df.index.year
        sim_df['month'] = sim_df.index.month
        sim_df["type"] = obs_type
        sim_df['type_time'] = (
            sim_df['type'] + "-" +
            sim_df['year'].astype(str) + "-" + sim_df['month'].astype(str)
            )
    if sim_ts == 3 and obs_ts == "month":
        sim_df = extract_mon_stf()
    # obs related
    obs_df = load_obs(obs_file, obs_ts, rch_num, obs_type)
    # combine
    com_so_df = sim_df.merge(obs_df, how='inner', on='type_time')
    com_so_df.to_csv("sim_obd.csv", index=False, float_format='%.7e')
    return com_so_df

def load_obs(obs_file, obs_ts, rch_num, obs_type):

    if obs_ts=="month":
        timestep = "monthly"
    obs_infile = "{}_{}{}.csv".format(obs_file, timestep, rch_num)
    obs_df = pd.read_csv(os.path.join("..", obs_infile))
    obs_df = obs_df.replace(-999, np.nan)
    obs_df["type"] = obs_type
    obs_df['type_time'] =(
        obs_df['type'] + "-" +
        obs_df['Year'].astype(str) + "-" + obs_df['Month'].astype(str)
    )
    obs_df = obs_df.loc[:, [obs_type, 'type_time']]
    obs_df.rename(columns={obs_type: "obd"}, inplace=True)
    return obs_df

    
# stf discharge
def extract_day_stf(rch_num, cali_stdate, cali_eddate):
    rch_file = rch_nam()
    stdate_, eddate_, sim_ts = get_start_end_step()
    channels = [rch_num]
    # FIXME: import APEXCON.read() causes error
    start_day = stdate_
    for i in channels:
        sim_stf = pd.read_csv(
                        rch_file,
                        delim_whitespace=True,
                        skiprows=9,
                        usecols=[0, 1, 8],
                        names=["idx", "sub", "sim"],
                        index_col=0)
        sim_stf = sim_stf.loc["REACH"]
        sim_stf_f = sim_stf.loc[sim_stf["sub"] == int(i)]
        sim_stf_f = sim_stf_f.drop(['sub'], axis=1)
        sim_stf_f.index = pd.date_range(start_day, periods=len(sim_stf_f.sim))
        sim_stf_f = sim_stf_f[cali_stdate:cali_eddate]
    return sim_stf_f

def extract_mon_stf(wd):
    os.chdir(wd)
    # APEXRUN.read()
    # APEXCONT.read()
    rch_file = parm.APEXRun_name + '.RCH'
    channels = parm.apex_outlets
    # start_day = "{}/{}/{}".format(parm.txt_ida, parm.txt_imo, parm.txt_iyr)
    # FIXME: import APEXCON.read() causes error
    start_day = "{}/{}/{}".format(1, 1, 1998)
    cali_start_day = parm.start_cal
    cali_end_day = parm.end_cal
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
        sim_stf_f.index = pd.date_range(start_day, periods=len(sim_stf_f.str_sim), freq='M')
        sim_stf_f = sim_stf_f[cali_start_day:cali_end_day]
    return sim_stf_f


def get_start_end_step():
    if os.path.isfile("APEXCONT.DAT"):
        with open('APEXCONT.DAT', "r") as f:
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

def rch_nam():
    with open("APEXRUN.DAT", "r") as f:
        data = [x.strip().split() for x in f]
        rch_nam = data[0][0] + ".RCH"
    return rch_nam


def com_sim_obd(self):
    sim_df = self.extract_day_stf()
    sim_df = sim_df.resample('M').mean()
    sim_df['year'] = sim_df.index.year
    sim_df['month'] = sim_df.index.month
    sim_df['time'] = sim_df['year'].astype(str) + "-" + sim_df['month'].astype(str)
    obs_df = self.load_obs()
    com_so_df = sim_df.merge(obs_df, how='inner', on='time')
    return com_so_df

# def get_rch_cols():

