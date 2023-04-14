import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import scipy.stats as stats
from ua.objectivefunctions import bias, rsquared, rmse
from matplotlib.pyplot import cm
import matplotlib.patches as mpatches
# from statistics import mean

def plot_one_one(df, numcols=1, fsize=8, ):
    fig1, ax = plt.subplots(figsize=(6,5))
    colors = cm.tab20(np.linspace(0, 1, len(df["type"].unique())))
    fmax = df.loc[:, ["sim", "obd"]].max().max()
    fmin = df.loc[:, ["sim", "obd"]].min().min()
    x_val = df.loc[:, "sim"].tolist()
    y_val = df.loc[:, "obd"].tolist()
    correlation_matrix = np.corrcoef(x_val, y_val)
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2
    m, b = np.polyfit(x_val, y_val, 1)
    ax.plot(np.array(x_val), (m*np.array(x_val)) + b, 'k', label='_nolegend_')
    ax.text(
            0.05, 0.9,
            f'$R^2:$ {r_squared:.3f}',
            horizontalalignment='left',
            bbox=dict(facecolor='gray', alpha=0.2),
            transform=ax.transAxes
            )
    ax.text(
            0.95, 0.05,
            f'$y={m:.2f}x{b:.2f}$',
            horizontalalignment='right',
            # bbox=dict(facecolor='gray', alpha=0.2),
            transform=ax.transAxes
            )
    lgds = []
    for tn, c in zip(df["type"].unique(), colors):
        sdf = df.loc[df['type'] == tn]
        ax.scatter(
            sdf.sim, sdf.obd, 
            color = c, 
            alpha=0.7)
        rsq_val = round(rsquared(sdf.obd, sdf.sim), 3)
        rmse_val = round(rmse(sdf.obd, sdf.sim), 3)
        lgds.append(f"{tn} (rsq:{rsq_val}, rmse:{rmse_val})")
    ax.plot([fmin, fmax], [fmin, fmax], 'k--', alpha=0.2)
    ax.set_xlabel("Simulated")
    ax.set_ylabel("Observed")
    fig1.legend(
        lgds, 
        bbox_to_anchor=(1.05, 1.05), 
        # ncols=numcols, 
        fontsize=fsize)
    # fig.tight_layout()
    fig1.savefig("plot_oneToOne.jpg", dpi=300, bbox_inches="tight")
    fig1.show()


def plot_senstivity_KDE(sa_df, num_sen_pars=10, fontsize=10, sen_type='ST', fig_name="sens_KDE.jpg", legend=True):
    """
    KDE: Kernel density estimation (KDE) to compute an empirical distribution of the sample
    """
    cols = [i[0] for i in sa_df.columns]
    unique_site = []
    for c in cols:
        if c not in unique_site:
            unique_site.append(c) 
    data = sa_df.loc[:, (unique_site, [sen_type]*len(unique_site))].T
    data = data.reindex(data.mean().sort_values(ascending=False).index, axis=1)
    # Boxplot
    f, ax = plt.subplots(figsize=(10,5))
    # plot. Set color of marker edge
    r = ax.violinplot(
        data,  widths=0.7, showmeans=True, showextrema=True,
        # bw_method='silverman'
        )
    sen_cols = num_sen_pars
    unsen_cols = len(data.columns) - sen_cols
    barcals = [u"#1f77b4"]*sen_cols + ["gray"]*unsen_cols
    meancals = [u"r"]*sen_cols + ["gray"]*unsen_cols
    r['cmeans'].set_color(meancals)
    r['cbars'].set_color(barcals)
    r['cmaxes'].set_color(barcals)
    r['cmins'].set_color(barcals)
    for i in r["bodies"][sen_cols:]:
        i.set_facecolor("gray")
    xlabels =  data.columns
    ax.set_xticks([i+1 for i in range(len(xlabels))])
    ax.set_xticklabels(xlabels, rotation=90)
    
    ftsize = fontsize
    for j in range(sen_cols, sen_cols+unsen_cols):
        ax.get_xticklabels()[j].set_color("gray")
    ax.tick_params(axis='both', labelsize=ftsize)
    if sen_type == "ST":
        ax.set_ylabel("Total SI", fontsize=ftsize)
    elif sen_type == "S1":
        ax.set_ylabel("First-Order SI", fontsize=ftsize)
    ax.set_xlabel("Parameter Names", fontsize=ftsize)
    if legend is True:
        reg_patch = mpatches.Patch(color=u"#1f77b4", label='Sensitive', alpha=0.5)
        gray_patch = mpatches.Patch(color=u"gray", label='Insensitive', alpha=0.5)
        plt.legend(handles=[reg_patch, gray_patch])
    plt.savefig(fig_name, dpi=300, bbox_inches="tight")
    plt.show()


def plot_sensitivity_scatter(
    sa_df, sen_type='ST', size=10, alpha=0.9,numcols=1, ftsize=10, fig_name="sens_scatter.jpg"
    ):
    cols = [i[0] for i in sa_df.columns]
    unique_site = []
    for c in cols:
        if c not in unique_site:
            unique_site.append(c) 
    data = sa_df.loc[:, (unique_site, [sen_type]*len(unique_site))]
    data = data.reindex(data.mean(axis=1).sort_values(ascending=False).index, axis=0)
    fig, ax = plt.subplots(figsize=(8,5))
    colors = cm.tab20(np.linspace(0, 1, len(data.columns)))
    lgds = []
    for tn, c in enumerate(colors):
        ax.scatter(
            data.index, data.iloc[:, tn],
            s=size,
            color=c, 
            alpha=alpha)
        lgds.append(data.columns[tn][0])
    ax.set_xticks([i for i in range(len(data))])
    ax.set_xticklabels(data.index, rotation=90)
    
    ax.tick_params(axis='both', labelsize=ftsize)
    if sen_type == "ST":
        ax.set_ylabel("Total SI", fontsize=ftsize)
    elif sen_type == "S1":
        ax.set_ylabel("First-Order SI", fontsize=ftsize)
    ax.set_xlabel("Parameter Names", fontsize=ftsize)

    plt.legend(
        lgds, 
        bbox_to_anchor=(1.01, 1.02), ncols=numcols)
    plt.savefig(fig_name, dpi=300, bbox_inches="tight")
    plt.show()


def equation(a, b):
    """Return a 1D polynomial."""
    return np.polyval(a, b) 


def plot_one_one2(df, numcols=1, fsize=8, ):
    fig, ax = plt.subplots(figsize=(6,5))
    colors = cm.rainbow(np.linspace(0, 1, len(df["type"].unique())))
    fmax = df.loc[:, ["somsc_sim", "obd"]].max().max()
    fmin = df.loc[:, ["somsc_sim", "obd"]].min().min()
    x_val = df.loc[:, "somsc_sim"].tolist()
    y_val = df.loc[:, "obd"].tolist()
    correlation_matrix = np.corrcoef(x_val, y_val)
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2
    p, cov = np.polyfit(x_val, y_val, 1, cov=True)
    y_model = np.polyval(p, x_val)

    # ax.plot(np.array(x_val), (m*np.array(x_val)) + b, 'k', label='_nolegend_')
    # Statistics
    n = len(y_val)                               # number of observations
    m = p.size                                                 # number of parameters
    m = 2                                                 # number of parameters
    dof = n - m                                                # degrees of freedom
    t = stats.t.ppf(0.975, n - m)                              # t-statistic; used for CI and PI bands    
    
    # Estimates of Error in Data/Model
    resid = y_val - y_model                      # residuals; diff. actual data from predicted values
    chi2 = np.sum((resid / y_model)**2)                        # chi-squared; estimates error in data
    chi2_red = chi2 / dof                                      # reduced chi-squared; measures goodness of fit
    s_err = np.sqrt(np.sum(resid**2) / dof)                    # standard deviation of the error
    
    # Fit
    ax.plot(x_val, y_model, "-", color="0.1", linewidth=1.5, alpha=0.5, label="Fit")  
    x2 = np.linspace(min(x_val), max(x_val), 100)
    y2 = equation(p, x2)

    # Confidence Interval (select one)
    plot_ci_manual(t, s_err, n, x_val, x2, y2, ax=ax)
    # plot_ci_bootstrap(x_val, y_val, resid, ax=ax)
    
    # Prediction Interval
    pi = t * s_err * np.sqrt(1 + 1/n + (x2 - np.mean(x_val))**2 / np.sum((x_val - np.mean(x_val))**2))   
    ax.fill_between(x2, y2 + pi, y2 - pi, color="None", linestyle="--")
    ax.plot(x2, y2 - pi, "--", color="0.5", label="95% Prediction Limits")
    ax.plot(x2, y2 + pi, "--", color="0.5")
    ax.text(
            0.05, 0.9,
            f'$R^2:$ {r_squared:.3f}',
            horizontalalignment='left',
            bbox=dict(facecolor='gray', alpha=0.8),
            transform=ax.transAxes
            )
    lgds = []
    for tn, c in zip(df["type"].unique(), colors):
        sdf = df.loc[df['site_name'] == tn]
        ax.scatter(
            sdf.somsc_sim, sdf.obd, 
            color = c, 
            alpha=0.7)
        rsq_val = round(rsquared(sdf.obd, sdf.somsc_sim), 3)
        rmse_val = round(rmse(sdf.obd, sdf.somsc_sim), 3)
        lgds.append(f"{tn} (rsq:{rsq_val}, rmse:{rmse_val})")
    ax.plot([fmin, fmax], [fmin, fmax], 'k--', alpha=0.2)
    ax.set_xlabel("Simulated SOC $(gCm^{-2})$")
    ax.set_ylabel("Observed SOC $(gCm^{-2})$")
    plt.legend(
        lgds, 
        bbox_to_anchor=(1.05, 1.05), ncols=numcols, fontsize=fsize)
    # fig.tight_layout()
    plt.savefig("plot_oneToOne.jpg", dpi=300, bbox_inches="tight")
    plt.show()


def plot_ci_manual(t, s_err, n, x, x2, y2, ax=None):
    """Return an axes of confidence bands using a simple approach.
    
    Notes
    -----
    .. math:: \left| \: \hat{\mu}_{y|x0} - \mu_{y|x0} \: \right| \; \leq \; T_{n-2}^{.975} \; \hat{\sigma} \; \sqrt{\frac{1}{n}+\frac{(x_0-\bar{x})^2}{\sum_{i=1}^n{(x_i-\bar{x})^2}}}
    .. math:: \hat{\sigma} = \sqrt{\sum_{i=1}^n{\frac{(y_i-\hat{y})^2}{n-2}}}
    
    References
    ----------
    .. [1] M. Duarte.  "Curve fitting," Jupyter Notebook.
       http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/CurveFitting.ipynb
    
    """
    if ax is None:
        ax = plt.gca()
    ci = t * s_err * np.sqrt(1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    ax.fill_between(x2, y2 + ci, y2 - ci, color="#b9cfe7", 
                    # edgecolor=""
                    )
    return ax


def plot_ci_bootstrap(xs, ys, resid, nboot=500, ax=None):
    """Return an axes of confidence bands using a bootstrap approach.

    Notes
    -----
    The bootstrap approach iteratively resampling residuals.
    It plots `nboot` number of straight lines and outlines the shape of a band.
    The density of overlapping lines indicates improved confidence.

    Returns
    -------
    ax : axes
        - Cluster of lines
        - Upper and Lower bounds (high and low) (optional)  Note: sensitive to outliers

    References
    ----------
    .. [1] J. Stults. "Visualizing Confidence Intervals", Various Consequences.
       http://www.variousconsequences.com/2010/02/visualizing-confidence-intervals.html

    """ 
    if ax is None:
        ax = plt.gca()

    bootindex = np.random.randint

    for _ in range(nboot):
        resamp_resid = resid[bootindex(0, len(resid) - 1, len(resid))]
        # Make coeffs of for polys
        pc = np.polyfit(xs, ys + resamp_resid, 1)                   
        # Plot bootstrap cluster
        ax.plot(xs, np.polyval(pc, xs), "b-", linewidth=2, alpha=3.0 / float(nboot))
    return ax


def plot_one_one_org(df, fsize=8):
    fig, ax = plt.subplots(figsize=(6,5))
    colors = cm.rainbow(np.linspace(0, 1, len(df["type"].unique())))
    lgds = []
    for tn, c in zip(df["type"].unique(), colors):
        sdf = df.loc[df['site_name'] == tn]
        ax.scatter(
            sdf.somsc_sim, sdf.obd, 
            color = c, 
            alpha=0.7)
        rsq_val = round(rsquared(sdf.obd, sdf.somsc_sim), 3)
        rmse_val = round(rmse(sdf.obd, sdf.somsc_sim), 3)
        lgds.append(f"{tn} (rsq:{rsq_val}, rmse:{rmse_val})")
    ax.set_xlabel("Simulated")
    ax.set_ylabel("Observed")
    plt.legend(
        lgds, 
        bbox_to_anchor=(1, 1), ncols=1, fontsize=fsize)
    # fig.tight_layout()
    plt.savefig("plot_oneToOne.jpg", dpi=300, bbox_inches="tight")
    plt.show()


def get_sensitivity_of_fast(results, like_index=1, M=4, print_to_console=True):
    """
    *** this function is originated from spotpy and modified for K-Fold SA
    Get the sensitivity for every parameter of your result array, created with the FAST algorithm

    :results: Expects an numpy array which should have as first axis an index "like" or "like1".
    :type: array

    :like_index: Optional, index of objectivefunction to base the sensitivity on, default=None first objectivefunction is taken
    :type: int

    :return: Sensitivity indices for every parameter
    :rtype: list
    """
    import math

    likes = results["like" + str(like_index)]
    # print("Number of model runs:", likes.size)
    parnames = get_parameternames(results)
    parnumber = len(parnames)
    # print("Number of parameters:", parnumber)

    rest = likes.size % (parnumber)
    if rest != 0:
        print(
            """"
            Number of samples in model output file must be a multiple of D,
            where D is the number of parameters in your parameter file.
          We handle this by ignoring the last """,
            rest,
            """runs.""",
        )
        likes = likes[:-rest]
    N = int(likes.size / parnumber)

    # Recreate the vector omega used in the sampling
    omega = np.zeros([parnumber])
    omega[0] = math.floor((N - 1) / (2 * M))
    m = math.floor(omega[0] / (2 * M))

    # print("m =", m)
    if m >= (parnumber - 1):
        omega[1:] = np.floor(np.linspace(1, m, parnumber - 1))
    else:
        omega[1:] = np.arange(parnumber - 1) % m + 1
    # print("Omega =", omega)
    # Calculate and Output the First and Total Order Values
    if print_to_console:
        print("Parameter First Total")
    Si = dict((k, [None] * parnumber) for k in ["S1", "ST"])
    # print(Si)
    for i in range(parnumber):
        l = np.arange(i * N, (i + 1) * N)
        # print(l)
        Si["S1"][i] = _compute_first_order(likes[l], N, M, omega[0])
        Si["ST"][i] = _compute_total_order(likes[l], N, omega[0])
        # print(Si)
        if print_to_console:
            print("%s %f %f" % (parnames[i], Si["S1"][i], Si["ST"][i]))
    return Si

def get_parameternames(results):
    """
    Get list of strings with the names of the parameters of your model.

    :results: Expects an numpy array which should have indices beginning with "par"
    :type: array

    :return: Strings with the names of the analysed parameters
    :rtype: list

    """
    fields = [word for word in results.dtype.names if word.startswith("par")]

    parnames = []
    for field in fields:
        parnames.append(field[3:])
    return parnames

def _compute_first_order(outputs, N, M, omega):
    f = np.fft.fft(outputs)
    Sp = np.power(np.absolute(f[np.arange(1, int((N + 1) / 2))]) / N, 2)
    V = 2 * np.sum(Sp)
    D1 = 2 * np.sum(Sp[np.arange(1, M + 1) * int(omega) - 1])
    return D1 / V


def _compute_total_order(outputs, N, omega):
    f = np.fft.fft(outputs)
    Sp = np.power(np.absolute(f[np.arange(1, int((N + 1) / 2))]) / N, 2)
    V = 2 * np.sum(Sp)
    Dt = 2 * sum(Sp[np.arange(int(omega / 2))])
    return 1 - Dt / V


class KFoldAnalyzer(object):
    # let's get header

    def __init__(self, proj_dir):
        self.proj_dir = proj_dir
        self.main_dir = os.path.join(self.proj_dir, "multi_main")
        os.chdir(self.proj_dir)

    def par_info(self):
        par_df = pd.read_csv('selected_pars.csv')
        par_df = par_df.loc[par_df['select']==1]
        par_df.reset_index(drop=True, inplace=True)
        return par_df
        
    # NOTE: we should get only directories
    def get_sites(self):
        # return os.listdir(self.main_dir)
        return next(os.walk(self.main_dir))[1]

    def sim_name(self):
        df = pd.read_csv(os.path.join(self.proj_dir, "k_fold_con.csv"))
        df['site_treat_time'] =df['treat_name'] +"_" +df['time'].astype(str)
        return df['site_treat_time'].tolist()


    def get_result_with_header(self, file_path):
        # parameter names
        par_list = self.par_info()['name'].tolist()
        sim_list = self.sim_name()
        tot_list = ["like1"] + par_list + sim_list + ["chain"]
        org_df = pd.read_csv(file_path, header=None)
        org_df.loc[-1] = tot_list
        org_df.index = org_df.index + 1
        org_df.sort_index(inplace=True)
        org_df.columns = org_df.iloc[0]
        org_df = org_df[1:]
        org_df.sort_index(inplace=True)
        return org_df

    
    def create_kfold_sims_pars(self, file_path, fold_number):
        print("Creating CSV files ...")
        kfold_df = pd.read_csv(os.path.join(self.proj_dir, "k_fold_con.csv"))
        kfold_df['site_treat_time'] =kfold_df['treat_name'] +"_" +kfold_df['time'].astype(str)
        df_header = self.get_result_with_header(file_path)
        df_header.to_csv(f'fold{fold_number:02d}_results.csv', index=False)
        print(f"...'fold{fold_number:02d}_results.csv' file was created ... passed")
        dfr = df_header.loc[:, kfold_df['site_treat_time']].T
        dfr = dfr.add_prefix('sim_')
        dfr['site_treat_time'] = dfr.index
        dfinal = kfold_df.merge(dfr, how='inner', on='site_treat_time')
        dfinal.to_csv(f'fold{fold_number:02d}_sims.csv', index=False)
        print(f"...'fold{fold_number:02d}_sims.csv' file was created ... passed")
        # create pars
        par_list = ["like1"]+ self.par_info()['name'].tolist() + ["chain"]
        pars_df = df_header.loc[:, par_list]
        pars_df.to_csv(f'fold{fold_number:02d}_pars.csv', index=False)
        print(f"...'fold{fold_number:02d}_pars.csv' file was created ... passed")
        print("Creating CSV files ... finished")

    def generate_joint_pars(self, posterior_num=100):
        pars_df = pd.DataFrame()
        for f in range(1, 6): # fixed to 5 folds
            df = pd.read_csv(f"fold{f:02d}_pars.csv")
            pars_df[f"fold{f:02d}_min"] = df.iloc[-posterior_num:, :].min(axis=0)
            pars_df[f"fold{f:02d}_max"] = df.iloc[-posterior_num:, :].max(axis=0)
        pars_df['join_min'] = pars_df.filter(regex='_min').min(axis=1)
        pars_df['join_max'] = pars_df.filter(regex='_max').max(axis=1)
        pars_df.to_csv("join_pars.csv")
        print(f"join_pars.csv' file was created ... passed")


def load_csv_par_result(filename):
    df = pd.read_csv(filename)
    return df

def load_csv_sim_result(filename):
    df = pd.read_csv(filename)
    return df

def load_csv_pair_result(filename):
    df = pd.read_csv(filename)
    return df


def plot_prior_posterior_par_hist(ax, results, par_df, par_id, prior_num, posterior_num):
    """plot prior and posterior parameter histogram
       This functing is the last 100 runs

    Args:
        ax (ax): _description_
        results (dataframe): parameter result dataframe
        par_df (dataframe): parameter dataset
        par_id (index): index from length of parameter dataframe
    """
    
    ax.hist(
        results.loc[:prior_num, par_df.loc[par_id, 'name']].values,
        bins=np.linspace(par_df.loc[par_id, 'min'], par_df.loc[par_id, 'max'], 20),color = "gray", alpha=0.5, density=True
    )
    ax.hist(
        results.loc[-posterior_num:, par_df.loc[par_id, 'name']].values,
        bins=np.linspace(par_df.loc[par_id, 'min'], par_df.loc[par_id, 'max'], 20), alpha=0.5, density=True
    )
    ax.set_ylabel("Density")
    ax.set_xlim(par_df.loc[par_id, 'min'], par_df.loc[par_id, 'max'])


def plot_parameter_trace(ax, result, par_df, par_id):
    """plot the parameter settings for each run and chain

    Args:
        ax (ax): _description_
        results (dataframe): parameter result dataframe
        par_df (dataframe): parameter dataset
        par_id (index): index from length of parameter dataframe
    """
    for i in range(int(result["chain"].max())):
        index= np.where(result["chain"] == i)
        ax.plot(result.loc[index, par_df.loc[par_id, 'name']].values, ".", markersize=2)
    ax.set_ylabel(par_df.loc[par_id, 'name'])
    ax.set_ylim(par_df.loc[par_id, 'min'], par_df.loc[par_id, 'max']) 

def plot_parameter_results(par_df, par_results, fold_number, fig_h=25, fig_w=10, prior_num=100, posterior_num=100):
    fig, ax = plt.subplots(nrows=12, ncols=2)
    fig.set_figheight(fig_h)
    fig.set_figwidth(fig_w)
    # for par_id in range(5,11):
    for par_id in range(len(par_df)):
        plot_parameter_trace(ax[par_id][0], par_results, par_df, par_id)
        plot_prior_posterior_par_hist(ax[par_id][1], par_results, par_df, par_id, prior_num, posterior_num)
    ax[-1][0].set_xlabel("Iterations")
    ax[-1][1].set_xlabel("Parameter range")
    fig.tight_layout()
    fig.savefig(
        f"fold{fold_number:02d}_parameter_uncertainty", 
        bbox_inches="tight", dpi=300)
    plt.show()


def plot_predicitive_uncertainty(sim_results, fold_number, pair=False, fig_h=7, fig_w=18, prior_num=100, posterior_num=100):
    fold_num = f"fold{fold_number:02d}"
    if pair is True:
        pass
    else:
        sim_results = sim_results.loc[sim_results[fold_num]==0]
    sim_filtered = sim_results.filter(regex='sim_') # get only simulated columns
    sim_filtered = sim_filtered.T
    if pair is True:
        sim_filtered.columns = shorten_colnam(sim_results.loc[:, 'pair_name'].tolist())
    else:
        sim_filtered.columns = sim_results.loc[:, 'site_treat_time'].tolist()
    sim_posterior = sim_filtered.iloc[-posterior_num:, :]
    sim_prior = sim_filtered.iloc[:prior_num, :]  

    fig , ax = plt.subplots()
    fig.set_figheight(fig_h)
    fig.set_figwidth(fig_w)
    # ax = plt.subplot(1, 1, 1)

    q5s, q25s, q75s, q95s = [], [], [], []
    for field in sim_prior.columns:
        q5s.append(
            np.percentile(sim_prior.loc[:, field], 2.5)
        )  # ALl 100 runs after convergence
        q95s.append(
            np.percentile(sim_prior.loc[:, field], 97.5)
        )  # ALl 100 runs after convergence
    label_added = False
    for i, field in enumerate(sim_prior.columns):
        values_within = [x for x in sim_prior.loc[:, field] if q5s[i] <= x <= q95s[i]]
        if not label_added:
            ax.scatter(
                [field]*len(values_within), values_within, c='gray', s=20, alpha=0.2,
                label="Prior Predictive Uncertainty")
            label_added = True
        else:
            ax.scatter([field]*len(values_within), values_within, c='gray', s=20, alpha=0.2,)
    q5s, q25s, q75s, q95s = [], [], [], []
    for field in sim_posterior.columns:
        q5s.append(
            np.percentile(sim_posterior.loc[:, field], 2.5)
        )  # ALl 100 runs after convergence
        q95s.append(
            np.percentile(sim_posterior.loc[:, field], 97.5)
        )  # ALl 100 runs after convergence
    
    label_added = False
    for i, field in enumerate(sim_posterior.columns):
        values_within = [x for x in sim_posterior.loc[:, field] if q5s[i] <= x <= q95s[i]]
        if not label_added:
            ax.scatter(
                [field]*len(values_within), values_within, c='b', s=20, alpha=0.2, 
                label="Posterior Predictive Uncertainty")
            label_added = True
        else:
            ax.scatter(
                [field]*len(values_within), values_within, c='b', s=20, alpha=0.2)
            
    if pair is True:
        ax.scatter(
            sim_posterior.columns, sim_results['SOC_change'].tolist(), 
            color="red", label="Observed", s=30).set_facecolor("none")
    else:
        ax.scatter(
            sim_posterior.columns, sim_results['obd'].tolist(), 
            color="red", label="Observed", s=30).set_facecolor("none")    
    
    ax.set_ylabel(r"Modeled $\Delta$SOC (g m$^{-2}$)")
    ax.set_xlabel("Simulations")
    ax.tick_params(axis='x', which='major', labelsize=8)
    plt.xticks(rotation=90)
    # fig.tight_layout()
    ax.margins(x=0.01)
    ax.legend()
    fig.savefig(f"fold{fold_number:02d}_predictive_uncertainty", bbox_inches="tight", dpi=300)
    plt.show()    


def plot_predicitive_uncertainty_v(sim_results, fold_number, fig_h=18, fig_w=7, prior_num=100, posterior_num=100):
    fold_num = f"fold{fold_number:02d}"
    sim_results = sim_results.loc[sim_results[fold_num]==0]
    sim_filtered = sim_results.filter(regex='sim_') # get only simulated columns
    sim_filtered = sim_filtered.T
    sim_filtered.columns = sim_results.loc[:, 'site_treat_time'].tolist()
    sim_posterior = sim_filtered.iloc[-posterior_num:, :]
    sim_prior = sim_filtered.iloc[:prior_num, :]  

    fig , ax = plt.subplots()
    fig.set_figheight(fig_h)
    fig.set_figwidth(fig_w)
    # ax = plt.subplot(1, 1, 1)

    q5s, q25s, q75s, q95s = [], [], [], []
    for field in sim_prior.columns:
        q5s.append(
            np.percentile(sim_prior.loc[:, field], 2.5)
        )  # ALl 100 runs after convergence
        q95s.append(
            np.percentile(sim_prior.loc[:, field], 97.5)
        )  # ALl 100 runs after convergence
    label_added = False
    for i, field in enumerate(sim_prior.columns):
        values_within = [x for x in sim_prior.loc[:, field] if q5s[i] <= x <= q95s[i]]
        if not label_added:
            ax.scatter(
                values_within, [field]*len(values_within), c='gray', s=20, alpha=0.2,
                label="Prior Predictive Uncertainty")
            label_added = True
        else:
            ax.scatter(values_within, [field]*len(values_within), c='gray', s=20, alpha=0.2,)
    q5s, q25s, q75s, q95s = [], [], [], []
    for field in sim_posterior.columns:
        q5s.append(
            np.percentile(sim_posterior.loc[:, field], 2.5)
        )  # ALl 100 runs after convergence
        q95s.append(
            np.percentile(sim_posterior.loc[:, field], 97.5)
        )  # ALl 100 runs after convergence
    
    label_added = False
    for i, field in enumerate(sim_posterior.columns):
        values_within = [x for x in sim_posterior.loc[:, field] if q5s[i] <= x <= q95s[i]]
        if not label_added:
            ax.scatter(
                values_within, [field]*len(values_within), c='b', s=20, alpha=0.2, 
                label="Posterior Predictive Uncertainty")
            label_added = True
        else:
            ax.scatter(
                values_within, [field]*len(values_within), c='b', s=20, alpha=0.2)
    ax.scatter(
        sim_results['obd'].tolist(), sim_posterior.columns,
        color="red", label="Observed", s=30).set_facecolor("none")
    ax.set_xlabel("Soil organic carbon (g C m$^{-2}$)")
    ax.set_ylabel("Simulations")
    ax.tick_params(axis='y', which='major', labelsize=8)
    ax.invert_yaxis()
    # plt.xticks(rotation=90)
    # fig.tight_layout()
    ax.margins(y=0.01)
    ax.legend()
    fig.savefig(f"fold{fold_number:02d}_predictive_uncertainty_v", bbox_inches="tight", dpi=300)
    plt.show()    


def create_fold_pair_file(sim_results_df, fold_number, pair_obd_df):
    fold_num = f"fold{fold_number:02d}"
    sim_results_df = sim_results_df.loc[sim_results_df[fold_num]==0]
    new_df = pd.DataFrame()
    soc_changes, pair_names = [], []

    for i in range(len(pair_obd_df)):
        if pair_obd_df.loc[i, "treat_1"] in sim_results_df.loc[:, 'site_treat_time'].tolist():
            treat1 = sim_results_df.loc[sim_results_df['site_treat_time']==pair_obd_df.loc[i, "treat_1"]].filter(regex='sim_')
            treat2 = sim_results_df.loc[sim_results_df['site_treat_time']==pair_obd_df.loc[i, "treat_2"]].filter(regex='sim_')
            soc_changes.append(pair_obd_df.loc[i, "SOC_change"])
            pair_names.append(f"{pair_obd_df.loc[i, 'treat_1']}_{pair_obd_df.loc[i, 'treat_2']}")
            treat1.reset_index(inplace=True)
            treat2.reset_index(inplace=True)
            treat1.loc[treat1.index] - treat2.loc[treat2.index]
            new_df = pd.concat([new_df, treat1.loc[treat1.index] - treat2.loc[treat2.index]], axis=0, ignore_index=True)
    new_df = new_df.drop('index', axis=1)
    new_df.insert(loc=0, column='pair_name', value=pair_names)
    new_df.insert(loc=1, column='SOC_change', value=soc_changes)
    new_df.to_csv(f"fold{fold_number:02d}_pairs.csv", index=False)
    print(f"'fold{fold_number:02d}_pairs.csv' file was created ...")


def shorten_colnam(long_colnam_list):
    shortnam_list = []
    for longnam in long_colnam_list:
        nam = []
        for i in longnam.split("_"):
            if not i in nam:
                nam.append(i)
        shortnam = '_'.join(nam)
        shortnam_list.append(shortnam)
    return shortnam_list

def create_fold_obj(pair_results, fold_number, posterior_num=100):
    obj_df = pd.DataFrame(
        index=get_site_names_fr_fold(fold_number),
        columns=[
        "model_bias_mean", "model_bias_min", "model_bias_min_num",
        "rmse_mean", "rmse_min", "rmse_min_num",
        "rsquared_mean", "rsquared_max", "rsquared_max_num"
        ])
    for site in get_site_names_fr_fold(fold_number):
        df = pair_results[pair_results['pair_name'].str.contains(site)]
        onlysim = df.filter(regex='sim_')
        onlysim = onlysim.iloc[:, -posterior_num:]
        bias_list = []
        rsquared_list = []
        rmse_list = []
        for i in onlysim.columns:
            bias_list.append(bias(df['SOC_change'].tolist(), onlysim[i].tolist()))
            rsquared_list.append(rsquared(df['SOC_change'].tolist(), onlysim[i].tolist()))
            rmse_list.append(rmse(df['SOC_change'].tolist(), onlysim[i].tolist()))
            
        obj_df.loc[site, "model_bias_mean"] = mean(bias_list)
        obj_df.loc[site, "model_bias_min"] = min(bias_list, key=abs)
        obj_df.loc[site, "model_bias_min_num"] = bias_list.index(min(bias_list, key=abs))
        obj_df.loc[site, "rmse_mean"] = mean(rmse_list)
        obj_df.loc[site, "rmse_min"] = min(rmse_list, key=abs)
        obj_df.loc[site, "rmse_min_num"] = int(rmse_list.index(min(rmse_list, key=abs)))
        obj_df.loc[site, "rsquared_mean"] = mean(rsquared_list)
        obj_df.loc[site, "rsquared_max"] = max(rsquared_list)
        obj_df.loc[site, "rsquared_max_num"] = int(rsquared_list.index(max(rsquared_list)))
    obj_df.to_csv(f"fold{fold_number:02d}_pair_objs.csv")
    print(f"'fold{fold_number:02d}_pair_objs.csv' was created ...")
    return obj_df

def get_site_names_fr_fold(fold_number):
    kfold_df =pd.read_csv("k_fold_con.csv") 
    fold_num = f"fold{fold_number:02d}"
    kfold_df = kfold_df.loc[kfold_df[fold_num]==0]
    return kfold_df['site_name'].unique().tolist()



def pick_random_par_sets(result_df, nFinalSets=168):
    result_df["chain_id"] =  [int(i) for i in result_df["chain"]+1]
    nPerChain = nFinalSets / int(result_df["chain_id"].max())
    pick_df = pd.DataFrame()
    for cid in result_df["chain_id"].unique():
        sel_df = result_df.loc[result_df["chain_id"]==cid]
        sel_df = sel_df.sort_values(by=['like1'], ascending=False)
        pick_df = pd.concat([pick_df, sel_df.iloc[:int(nPerChain), :]], axis=0)
    pick_df.drop("chain", axis=1, inplace=True)
    pick_df.to_csv("ensemble_pars_sims.csv", index=False)
    return pick_df