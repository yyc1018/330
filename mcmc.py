import pandas as pd
import numpy as np
import stochasticBerg as SB
from pydream.parameters import SampledParam
from scipy.stats import loguniform
from pydream.core import run_dream
from pydream.convergence import Gelman_Rubin


########################################################################################################################
# This script is provided by my supervisor, Dr Ruben Perez-Carrasco
# I have made minor edits, such as loading the data, adding a transition matrix and zeta parameter for each bead
# diameter, adjusting the lower and max limits for the prior distributions, adding blank lines and renaming variables
# for readability, rephrasing comments, etc.
########################################################################################################################
# Time difference between two consecutive points, Delta_t and experimental stator stoichiometry trajectories

Delta_t = 0.1

# Load data
ssData = pd.read_csv("/Users/katherine.yychow/OneDrive - Imperial College London/Year 3/Final year "
                     "project/Code/Final Scripts/ssData.csv", index_col=0)
stallData = pd.read_csv("/Users/katherine.yychow/OneDrive - Imperial College London/Year 3/Final year "
                        "project/Code/Final Scripts/stallData.csv", index_col=0)
resurData = pd.read_csv("/Users/katherine.yychow/OneDrive - Imperial College London/Year 3/Final year "
                        "project/Code/Final Scripts/resurData.csv", index_col=0)

# Replace all NANs with zeros and 14 and 15 stator stoichiometry with 13 because Nmax = 13
ssData.iloc[:, : 4794] = ssData.iloc[:, : 4794].replace(np.nan, 0)
ssData.iloc[:, : 4794] = ssData.iloc[:, : 4794].replace([14, '14'], 13)
ssData.iloc[:, : 4794] = ssData.iloc[:, : 4794].replace([15, '15'], 13)
stallData.iloc[:, : 4794] = stallData.iloc[:, : 4794].replace(np.nan, 0)
stallData.iloc[:, : 4794] = stallData.iloc[:, : 4794].replace([14, '14'], 13)
stallData.iloc[:, : 4794] = stallData.iloc[:, : 4794].replace([15, '15'], 13)
resurData.iloc[:, : 4794] = resurData.iloc[:, : 4794].replace(np.nan, 0)
resurData.iloc[:, : 4794] = resurData.iloc[:, : 4794].replace([14, '14'], 13)
resurData.iloc[:, : 4794] = resurData.iloc[:, : 4794].replace([15, '15'], 13)

allData = [ssData.iloc[:, : 4794], stallData.iloc[:, : 4794], resurData.iloc[:, : 4794]]
trajectories = pd.concat(allData)
trajectories = trajectories.astype('int32')
trajectoryArrays = trajectories.values

ssData_D300 = ssData.iloc[0: 33, : 4794].astype('int32')
ssData_D500 = ssData.iloc[34: 63, : 4794].astype('int32')
ssData_D1300 = ssData.iloc[64: 92, : 4794].astype('int32')

stallData_D300 = stallData.iloc[0: 33, : 4794].astype('int32')
stallData_D500 = stallData.iloc[34: 63, : 4794].astype('int32')
stallData_D1300 = stallData.iloc[64: 92, : 4794].astype('int32')

resurData_D300 = resurData.iloc[0: 23, : 4794].astype('int32')
resurData_D500 = resurData.iloc[24: 45, : 4794].astype('int32')
resurData_D1300 = resurData.iloc[46: 65, : 4794].astype('int32')

# Trajectories for each bead diameter

trajectories_D300 = pd.concat((ssData_D300, stallData_D300, resurData_D300))
trajectoryArrays_D300 = trajectories_D300.values

trajectories_D500 = pd.concat((ssData_D500, stallData_D500, resurData_D500))
trajectoryArrays_D500 = trajectories_D500.values

trajectories_D1300 = pd.concat((ssData_D1300, stallData_D1300, resurData_D1300))
trajectoryArrays_D1300 = trajectories_D1300.values

########################################################################################################################
# Summary matrix containing the occurrences of each possible transition
N_max = 13
transitionMatrix_D300 = np.zeros((N_max + 1, N_max + 1))
transitionMatrix_D500 = np.zeros((N_max + 1, N_max + 1))
transitionMatrix_D1300 = np.zeros((N_max + 1, N_max + 1))
for trajectory in trajectoryArrays_D300:
    transitions_D300 = np.vstack((trajectory[1:], trajectory[:-1]))
    for couple in transitions_D300.T:
        transitionMatrix_D300[couple[0], couple[1]] += 1

for trajectory in trajectoryArrays_D500:
    transitions_D500 = np.vstack((trajectory[1:], trajectory[:-1]))
    for couple in transitions_D500.T:
        transitionMatrix_D500[couple[0], couple[1]] += 1

for trajectory in trajectoryArrays_D1300:
    transitions_D1300 = np.vstack((trajectory[1:], trajectory[:-1]))
    for couple in transitions_D1300.T:
        transitionMatrix_D1300[couple[0], couple[1]] += 1


########################################################################################################################
# Inference requires a likelihood that accepts the parameters to be explored as input, and returns the loglikelihood as
# the output

def loglikelihood_pars(parameters):
    k0, alpha, gamma_D300, gamma_D500, gamma_D1300, beta = parameters
    B_D300 = SB.Berg_Matrix(k0, alpha, gamma_D300)
    B_D500 = SB.Berg_Matrix(k0, alpha, gamma_D500)
    B_D1300 = SB.Berg_Matrix(k0, alpha, gamma_D1300)
    Q_D300, logQ_D300 = SB.GetTransitionMatrix(B_D300, Delta_t)
    Q_D500, logQ_D500 = SB.GetTransitionMatrix(B_D500, Delta_t)
    Q_D1300, logQ_D1300 = SB.GetTransitionMatrix(B_D1300, Delta_t)

    loglikelihoodList_D300 = 0
    loglikelihoodList_D500 = 0
    loglikelihoodList_D1300 = 0

    for trajectory300 in trajectoryArrays_D300:
        loglikelihoodList_D300 += SB.LogLikelihood(logQ_D300, trajectory300)

    for trajectory500 in trajectoryArrays_D500:
        loglikelihoodList_D500 += SB.LogLikelihood(logQ_D500, trajectory500)

    for trajectory1300 in trajectoryArrays_D1300:
        loglikelihoodList_D1300 += SB.LogLikelihood(logQ_D1300, trajectory1300)
    return np.sum(loglikelihoodList_D300), np.sum(loglikelihoodList_D500), np.sum(loglikelihoodList_D1300)


def loglikelihood_pars_occurences(parameters, noUncertainty=False):
    k0, alpha, gamma_D300, gamma_D500, gamma_D1300, beta = parameters
    B_D300 = SB.Berg_Matrix(k0, alpha, gamma_D300)
    B_D500 = SB.Berg_Matrix(k0, alpha, gamma_D500)
    B_D1300 = SB.Berg_Matrix(k0, alpha, gamma_D1300)

    if noUncertainty:
        Q_D300, logQ_D300 = SB.GetTransitionMatrix_Fast(B_D300, Delta_t)
        Q_D500, logQ_D500 = SB.GetTransitionMatrix_Fast(B_D500, Delta_t)
        Q_D1300, logQ_D1300 = SB.GetTransitionMatrix_Fast(B_D1300, Delta_t)

    else:
        preQ_D300, prelogQ_D300 = SB.GetTransitionMatrix_Fast(B_D300, Delta_t)
        preQ_D500, prelogQ_D500 = SB.GetTransitionMatrix_Fast(B_D500, Delta_t)
        preQ_D1300, prelogQ_D1300 = SB.GetTransitionMatrix_Fast(B_D1300, Delta_t)

        Q_D300, logQ_D300 = SB.TransitionMatrix_WithUncertainty(preQ_D300, beta)
        Q_D500, logQ_D500 = SB.TransitionMatrix_WithUncertainty(preQ_D500, beta)
        Q_D1300, logQ_D1300 = SB.TransitionMatrix_WithUncertainty(preQ_D1300, beta)


    return (SB.LogLikelihood_occurences(logQ_D300, transitionMatrix_D300) +
            SB.LogLikelihood_occurences(logQ_D500, transitionMatrix_D500) +
            SB.LogLikelihood_occurences(logQ_D1300, transitionMatrix_D1300))


########################################################################################################################
# Defining prior distributions for k0, alpha, the three gammas, and beta
lower_limits = [0.0002, 1, 0.02, 0.002, 0.002, 0.0001]  # Lower values for each parameter prior
max_limits = [0.001, 1000, 2, 2, 2, 1]  # Width of each parameter prior
prior_distributions = [SampledParam(loguniform, a=lower_limits, b=max_limits)]

########################################################################################################################
# Parameters necessary to run the MCMC
niterations = 5000  # Number of iterations between checking for convergence
converged = False
nchains = 5  # Number of parallel chains
GRlim = 1.2  # Gelman-Rubin statistic convergence limit
historythin = 100  # Storing every 100th iteration in the MCMC chain to reduce storage requirements
run_dream_flag = True

########################################################################################################################
# Initializing counters
total_iterations = 0
start = None  # Start of each chain is chosen to be random
restart = False  # Use previous chain (set to True when not converging)
cumulative_samples = None

########################################################################################################################
# Run of pyDREAM
if __name__ == '__main__':
    while not converged:
        sampled_params, log_ps = run_dream(prior_distributions, loglikelihood_pars_occurences,
                                           niterations=niterations, nchains=nchains,
                                           multitry=True, gamma_levels=4, adapt_gamma=True,
                                           parallel=True, history_thin=historythin, start=start, restart=restart,
                                           model_name='Berg_Model', verbose=True)

        # Save sampling output (sampled parameter values and their corresponding logps)
        for chain in range(len(sampled_params)):
            np.save('pydream_results/' + 'parapop_sampled_params_' + 'chain_' + str(chain), sampled_params[chain])
            np.save('pydream_results/' + 'parapop_logps_' + 'chain_' + str(chain), log_ps[chain])

        # Calculating convergence
        total_iterations += niterations
        if cumulative_samples is None:
            cumulative_samples = sampled_params
        else:
            cumulative_samples = [np.concatenate((cumulative_samples[chain], sampled_params[chain])) for chain in
                                  range(nchains)]
        GR = Gelman_Rubin(cumulative_samples)
        print('At iteration: ', total_iterations, ' GR = ', GR)
        np.savetxt('pydream_results/' + 'parapop_GelmanRubin_iteration_' + str(niterations) + '.out', GR)

        # Resetting new cycle
        if np.all(GR < GRlim):  # All chains need to converge,
            converged = True
        else:  # otherwise the results are stored and the MCMC is repeated
            start = [sampled_params[chain][-1, :] for chain in range(nchains)]
            restart = True
