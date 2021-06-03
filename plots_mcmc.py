import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import scipy.stats
import pymc3

########################################################################################################################
# Load parameter values sampled from MCMC's five parallel chains
# Columns are parameters and rows are iterations
results1 = np.load('/Users/katherine.yychow/Downloads/pydream_results 2/parapop_sampled_params_chain_0.npy')
results2 = np.load('/Users/katherine.yychow/Downloads/pydream_results 2/parapop_sampled_params_chain_1.npy')
results3 = np.load('/Users/katherine.yychow/Downloads/pydream_results 2/parapop_sampled_params_chain_2.npy')
results4 = np.load('/Users/katherine.yychow/Downloads/pydream_results 2/parapop_sampled_params_chain_3.npy')
results5 = np.load('/Users/katherine.yychow/Downloads/pydream_results 2/parapop_sampled_params_chain_4.npy')

paramData = np.vstack((results1, results2, results3, results4, results5))

########################################################################################################################
# Determine burn-in period
fig, axes = plt.subplots(5, 1)
for i in range(5):
    sns.lineplot(ax=axes[i], data=paramData[0: 5000, i], linewidth=0.5)
    sns.lineplot(ax=axes[i], data=paramData[5000: 10000, i], linewidth=0.5)
    sns.lineplot(ax=axes[i], data=paramData[10000: 15000, i], linewidth=0.5)
    sns.lineplot(ax=axes[i], data=paramData[15000: 20000, i], linewidth=0.5)
    sns.lineplot(ax=axes[i], data=paramData[20000: 25000, i], linewidth=0.5)

axes[0].set_ylabel('k0 (/s)')
axes[1].set_ylabel('\u03B1')
axes[2].set_ylabel('\u03B6300')
axes[3].set_ylabel('\u03B6500')
axes[4].set_ylabel('\u03B61300')
axes[4].set_xlabel('Iterations')

burnIn = 80

########################################################################################################################
# Remove burn-in period from sampled parameter values
paramData_burnIn = np.vstack((results1[burnIn: 25000], results2[burnIn: 25000], results3[burnIn: 25000], results4[burnIn: 25000], results5[burnIn: 25000]))

# Log all parameter values and convert into a pandas DataFrame with column names
log_paramData_burnIn = np.log(paramData_burnIn)
log_paramVals_burnIn = pd.DataFrame(data=log_paramData_burnIn)
log_paramVals_burnIn.columns = ["log(k0) (/s)", "log(\u03B1)", "log(\u03B6300)", "log(\u03B6500)", "log(\u03B61300)"]
paramNames = ["log(k0) (/s)", "log(\u03B1)", "log(\u03B6300)", "log(\u03B6500)", "log(\u03B61300)"]

########################################################################################################################
# Create a subplot grid to plot the pairwise relationships between parameter values
BinfPlot = sns.PairGrid(log_paramVals_burnIn, vars=paramNames)
axes = BinfPlot.axes

# Diagonal contains distributions
BinfPlot.map_diag(sns.histplot)

# Upper off-diagonal contains scatter plots
BinfPlot.map_upper(sns.scatterplot, s=10)

# Lower off-diagonal contains KDE plots
BinfPlot.map_lower(sns.kdeplot)

plt.show()

########################################################################################################################
# Calculate 95% high density region
for i in range(5):
    print('HDI lower and upper: ', pymc3.stats.hpd(paramData_burnIn[:, i]))


# Calculate mean and 95% confidence interval
def mean_confidenceInterval(paramVals, confidence=0.95):  # Adapted from user shasan's answer to "Compute a confidence
    # interval from sample data". Available at:
    # https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    a = 1.0 * np.array(paramVals)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    print('mean: ', m, 'CI lower: ', m - h, 'CI upper: ', m + h)
    return m, m - h, m + h


for i in range(5):
    mean_confidenceInterval(paramData_burnIn[:, i])
