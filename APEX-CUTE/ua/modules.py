import os
# import spotpy
import pandas as pd
import numpy as np
from ua.models import APEX_setup
from ua.likelihoods import gaussianLikelihoodMeasErrorOut as GLMEOUT
from ua.likelihoods import gaussianLikelihoodHomoHeteroDataError as GLHHDE
from ua.algorithms import dream_ac


# def run_dream(ui):
#     APEX_setup(ui)

def run_dream(ui, eps=10e-6, nChains=10, 
        dbname="DREAM_apex", dbformat="csv", parallel='mpc', obj_func=GLHHDE):
    # spot_setup = single_setup(GausianLike)

    # Bayesian algorithms should be run with a likelihood function
    # obj_func = ua.likelihoods.gaussianLikelihoodHomoHeteroDataError
    # obj_func = spotpy.likelihoods.gaussianLikelihoodMeasErrorOut
    spot_setup = APEX_setup(ui, parallel=parallel, obj_func=obj_func)
    # Select seven chains and set the Gelman-Rubin convergence limit
    delta = 3
    convergence_limit = 1.2

    # Other possible settings to modify the DREAM algorithm, for details see Vrugt (2016)
    c = 0.1
    nCr = 3
    runs_after_convergence = 100
    acceptance_test_option = 6

    # sampler = spotpy.algorithms.dream(
    #     spot_setup, dbname=dbname, dbformat=dbformat, parallel=parallel,
    #     # dbappend=True
    #     )
    sampler = dream_ac.dream(
        spot_setup, dbname=dbname, dbformat=dbformat, parallel=parallel,
        # dbappend=True
        )
    r_hat = sampler.sample(
        ui,
        100,
        nChains,
        nCr,
        delta,
        c,
        eps,
        convergence_limit,
        runs_after_convergence,
        acceptance_test_option,
    )
    if dbformat == 'ram':
        results = pd.DataFrame(sampler.getdata())
        results.to_csv(f"{dbname}.csv", index=False)
        #########################################################
        # Example plot to show the convergence #################
        results02 = spotpy.analyser.load_csv_results(f"{dbname}")
        spotpy.analyser.plot_gelman_rubin(results02, r_hat, fig_name="DREAM_r_hat.png")