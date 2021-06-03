import pandas as pd
import numpy as np
import math as m
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import odeint

########################################################################################################################
# Load data
ssData = pd.read_csv('/Users/katherine.yychow/OneDrive - Imperial College London/Year 3/Final year project/Code/Final '
                     'Scripts/ssData.csv', index_col=0)

stallData = pd.read_csv('/Users/katherine.yychow/OneDrive - Imperial College London/Year 3/Final year '
                        'project/Code/Final Scripts/stallData.csv', index_col=0)

resurData = pd.read_csv('/Users/katherine.yychow/OneDrive - Imperial College London/Year 3/Final year '
                        'project/Code/Final Scripts/resurData.csv', index_col=0)

########################################################################################################################
# Replace all zeros in steady state and stall trajectories with NAN
ssData.iloc[:, : 4794] = ssData.iloc[:, : 4794].replace(['0', 0], np.nan)
stallData.iloc[:, : 8394] = stallData.iloc[:, : 8394].replace(['0', 0], np.nan)

# Replace zeroes after 200s in resurrection trajectories with NAN
resurData.iloc[:, 200: 15814] = resurData.iloc[:, 200: 15814].replace(['0', 0], np.nan)

########################################################################################################################
# Calculate mean of all trajectories
ssMean_D300 = ssData.iloc[0: 33].mean(axis=0)
ssMean_D500 = ssData.iloc[34: 63].mean(axis=0)
ssMean_D1300 = ssData.iloc[64: 92].mean(axis=0)

stallMean_D300 = stallData.iloc[0: 33].mean(axis=0)
stallMean_D500 = stallData.iloc[34: 63].mean(axis=0)
stallMean_D1300 = stallData.iloc[64: 92].mean(axis=0)

resurMean_D300 = resurData.iloc[0: 23].mean(axis=0)
resurMean_D500 = resurData.iloc[24: 45].mean(axis=0)
resurMean_D1300 = resurData.iloc[46: 65].mean(axis=0)


########################################################################################################################
# Define Berg model
def dNdtModel(N, t, k0, alpha, zeta):
    kon = k0 * (1 - m.e ** (-alpha / N))
    koff = kon * m.e ** zeta
    return kon * (13 - N) - koff * N


########################################################################################################################
# Range of time points for each of the three phases
t_ss = np.arange(-ssMean_D300.shape[0] / 10, 0, 0.1)
t_stall = np.arange(0, 839.4, 0.1)
t_resur = np.arange(839.4, 839.4 + 1581.4, 0.1)

# Steady state mean
ssGlobal_D300 = ssMean_D300.mean()
ssGlobal_D500 = ssMean_D500.mean()
ssGlobal_D1300 = ssMean_D1300.mean()

ssGlobalList = [ssGlobal_D300, ssGlobal_D500, ssGlobal_D1300]

# Range of time points for steady state mean
t_all = np.arange(-ssMean_D300.shape[0] / 10, 839.4 + 1581.4, 0.1)

########################################################################################################################
# Initial values
N0List = [6.18181818, 9.23333333, 10, 0.30434783, 0, 0]

# Results from least squares minimisation
k0 = 0.00890236
alpha = 2.11277947
zetaList = [0.62952840, -0.37997447, -1.23221807]

########################################################################################################################
# Plot
fig, axes = plt.subplots(3, 1)
sns.set_style("white")

for i in range(3):  # Steady state mean
    sns.lineplot(ax=axes[i], x=t_all, y=ssGlobalList[i], color='grey', linestyle='--', linewidth=2)

for i in range(92):  # All individual steady state trajectories
    if i <= 33:
        axes[0].plot(t_ss, ssData.iloc[i, :], color='grey', linestyle='--', linewidth=0.5)
    elif i <= 63:
        axes[1].plot(t_ss, ssData.iloc[i, :], color='grey', linestyle='--', linewidth=0.5)
    else:
        axes[2].plot(t_ss, ssData.iloc[i, :], color='grey', linestyle='--', linewidth=0.5)

for i in range(92):  # All individual stall trajectories
    if i <= 33:
        axes[0].plot(t_stall, stallData.iloc[i, :], color='grey', linestyle='--', linewidth=0.5)
    elif i <= 63:
        axes[1].plot(t_stall, stallData.iloc[i, :], color='grey', linestyle='--', linewidth=0.5)
    else:
        axes[2].plot(t_stall, stallData.iloc[i, :], color='grey', linestyle='--', linewidth=0.5)

for i in range(65):  # All individual resurrection trajectories
    if i <= 23:
        axes[0].plot(t_resur, resurData.iloc[i, :], color='grey', linestyle='--', linewidth=0.5)
    elif i <= 45:
        axes[1].plot(t_resur, resurData.iloc[i, :], color='grey', linestyle='--', linewidth=0.5)
    else:
        axes[2].plot(t_resur, resurData.iloc[i, :], color='grey', linestyle='--', linewidth=0.5)

# Mean trajectories for steady state, stall, and resurrection
sns.lineplot(ax=axes[0], x=t_ss, y=ssMean_D300, color='k', linewidth=2)
sns.lineplot(ax=axes[1], x=t_ss, y=ssMean_D500, color='k', linewidth=2)
sns.lineplot(ax=axes[2], x=t_ss, y=ssMean_D1300, color='k', linewidth=2)

sns.lineplot(ax=axes[0], x=t_stall, y=stallMean_D300, color='k', linewidth=2)
sns.lineplot(ax=axes[1], x=t_stall, y=stallMean_D500, color='k', linewidth=2)
sns.lineplot(ax=axes[2], x=t_stall, y=stallMean_D1300, color='k', linewidth=2)

sns.lineplot(ax=axes[0], x=t_resur, y=resurMean_D300, color='k', linewidth=2)
sns.lineplot(ax=axes[1], x=t_resur, y=resurMean_D500, color='k', linewidth=2)
sns.lineplot(ax=axes[2], x=t_resur, y=resurMean_D1300, color='k', linewidth=2)

# Fitted curves as per the least squares minimisation results
for i in range(3):
    axes[i].plot(t_stall, (odeint(dNdtModel, N0List[i], t_stall, args=(k0, alpha, zetaList[i]))), color='r',
                 linewidth=2)
    axes[i].plot(t_resur, (odeint(dNdtModel, N0List[i + 3], t_resur, args=(k0, alpha, zetaList[i]))), color='r',
                 linewidth=2)
    axes[i].set_xlim(-ssMean_D300.shape[0] / 10, 1700)
    axes[i].set_ylabel('N')
    axes[i].set_xlabel('t (s)')
    axes[i].set_ylim(0, 13)

axes[0].set_title('300nm')
axes[1].set_title('500nm')
axes[2].set_title('1300nm')

plt.show()
