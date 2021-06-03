# 330
Statistical modelling of the dynamics of the Bacterial Flagellar Motor

Provided are all experimental data, scripts, and Bayesian inference result files used in the project. All files are written by me (Yui Yee Chow) unless otherwise stated.

Below is a brief description of each file's contents in the order that they were used.

- allData: contains a WeTransfer link to all experimental data

- experimentalData.py and mean_experimentalData.py: extracts the stator stoichiometry trajectories from experimentalData files

- leastSquares.py

- plots_leastSquares.py

- stochasticBerg.py: functions required to run MCMC. Written by my supervisor, Dr Ruben Perez-Carrasco

- mcmc.py: PyDREAM implementation of MCMC. Written by my supervisor, Dr Ruben Perez-Carrasco

- pydream_results: contains a WeTransfer link to a file containing all MCMC results. MCMC results from each run are stored in folders numbered from 2 to 5. Each of these folders contain parapop_sampled_params_chain_i.npy files where i = 1 to 5 for each of the 5 chains.

- plots_mcmc.py: required to plot pydream_results 2

- plots_mcmc_uncertainty.py: required to plot pydream_results 3, 4, and 5
