import pandas as pd
import numpy as np
import math as m
from scipy.integrate import odeint
from lmfit import Parameters, minimize, report_fit


########################################################################################################################
# Load data
ssMean = pd.read_csv('/Users/katherine.yychow/OneDrive - Imperial College London/Year 3/Final year project/Code/Final '
                     'Scripts/mean_ssData.csv', index_col=0).to_numpy()

stallMean = pd.read_csv('/Users/katherine.yychow/OneDrive - Imperial College London/Year 3/Final year '
                        'project/Code/Final Scripts/mean_stallData.csv', index_col=0).to_numpy()

resurMean = pd.read_csv('/Users/katherine.yychow/OneDrive - Imperial College London/Year 3/Final year '
                        'project/Code/Final Scripts/mean_resurData.csv', index_col=0).to_numpy()


# Data with contents by row being mean stall 300nm, 500nm, 1300nm, then mean resurrection 300nm, 500nm and 1300nm
stallresurData = np.stack((stallMean[0][0:270], stallMean[1][0:270], stallMean[2][0:270], resurMean[0][0:270], resurMean[1][0:270], resurMean[2][0:270]))

########################################################################################################################
# First data point as the initial conditions
N0_stallD300 = 6.18181818
N0_stallD500 = 9.23333333
N0_stallD1300 = 10
N0_resurD300 = 0.30434783
N0_resurD500 = 0
N0_resurD1300 = 0

N0 = [N0_stallD300, N0_stallD500, N0_stallD1300, N0_resurD300, N0_resurD500, N0_resurD1300]

t = np.arange(0, 270, 1)

########################################################################################################################
# Define Berg model
def dNdtModel(N, t, k0, alpha, zeta):
    kon = k0 * (1 - m.e ** (-alpha / N))
    koff = kon * m.e ** zeta
    return kon * (13 - N) - koff * N


# Solve the Berg model
def dNdtSol(t, N0, params, i):
    k0 = params['k0_1'].value
    alpha = params['alpha_1'].value
    if i == 0:
        zeta = params['zeta_1'].value
    elif i == 1:
        zeta = params['zeta_2'].value
    else:
        zeta = params['zeta_3'].value
    sol = odeint(dNdtModel, N0, t, args=(k0, alpha, zeta))
    return sol


dataShape, _ = stallresurData.shape


# Residuals function
def residuals(params, N0, t, stallresurData):
    resid = 0 * stallresurData[:]

    for i in range(dataShape):
        resid[i, :] = stallresurData[i, :] - np.transpose(dNdtSol(t, N0[i], params, i))
    return resid.flatten()


########################################################################################################################
# Least squares minimisation parameter fits from an internal paper titled "Internal: In Favor of Berg’s model?" as
# initial estimates. This paper was provided by my supervisor, Dr Ruben Perez-Carrasco, and no author was listed.
# Varying these estimates has little effect on the results
k0 = 0.0081
alpha = (17.2402 + 3.4480 + 1.5325) / 3
zeta = (0.7214, -0.4419, -1.3291, 0.7214, -0.4419, -1.3291)

########################################################################################################################
# Create six sets of parameters to estimate (three bead diameters * two phases: stall and resurrection)
params = Parameters()

for i in range(6):
    params.add('k0_%i' % (i + 1), value=k0, min=0, max=1)
    params.add('alpha_%i' % (i + 1), value=alpha, min=0, max=40)
    params.add('zeta_%i' % (i + 1), value=zeta[i], min=-5, max=5)

# Constrain the values of k0 and alpha to be the same for all data
for i in (2, 3, 4, 5, 6):
    params['k0_%i' % i].expr = 'k0_1'
    params['alpha_%i' % i].expr = 'alpha_1'

# Constrain the values of zeta to be the same for each bead diameter
params['zeta_4'].expr = 'zeta_1'
params['zeta_5'].expr = 'zeta_2'
params['zeta_6'].expr = 'zeta_3'

########################################################################################################################
# Print results
result = minimize(residuals, params, method='leastsq', args=(N0, t, stallresurData))
report_fit(result)
