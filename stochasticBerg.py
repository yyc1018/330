import numpy as np
from scipy import linalg


########################################################################################################################
# This script is provided by my supervisor, Dr Ruben Perez-Carrasco
# I have made minor edits, such as adding blank lines and renaming variables for readability, rephrasing comments, etc.
########################################################################################################################

errLog = 1E-15  # Used to mark loglikelihood too small


def Berg_Matrix(k0, alpha, gamma, N=13):  # Eq (7)
    # Return rate Matrix B for Berg's model so that dP/dt(n,t) = B P(n,t)
    # Number of sites is N, so matrix must be (N + 1, N + 1) to accommodate state 0 (redundant but clearer)

    B = np.zeros((N + 1, N + 1))
    idxs = np.arange(0, N)  # This vector will be used frequently for filling the matrix
    # For this model, the matrix is tri-diagonal. We will fill the matrix diagonal per diagonal
    # First diagonal contains off-rates
    diagonal_top = tuple(np.vstack((idxs, idxs + 1)))
    B[diagonal_top] = gamma * k0 * (1 - np.exp(-alpha / (idxs + 1))) * (idxs + 1)
    # Bottom diagonal contains on-rates
    diagonal_bottom = tuple(np.vstack((idxs[1:] + 1, idxs[1:])))  # [1:] because first element will be introduced
    # separately
    B[diagonal_bottom] = k0 * (1 - np.exp(-alpha / (idxs[1:]))) * (N - idxs[1:])
    B[1, 0] = k0 * N  # Introduced separately to avoid numeric error at exp(-alpha / N) when N->0
    # Middle diagonal
    diagonal_middle = tuple(np.vstack((idxs[1:], idxs[1:])))  # [1:-1] because first and last element will be
    # introduced separately
    B[diagonal_middle] = -k0 * ((N - idxs[1:]) + gamma * idxs[1:]) * (1 - np.exp(-alpha / (idxs[1:])))
    B[0, 0] = -k0 * N
    B[-1, -1] = -k0 * N * gamma * (1 - np.exp(-alpha / N))
    return B


def GetEigenElements(B):  # Text between eq (7) and  (9)
    # Takes Matrix B and computes the eigendecomposition and the change of basis matrix. This function is useful as
    # it does not have to be computed every time the function is integrated (eq. 8)
    eigen_l, U_neg = linalg.eig(B)  # Diagonalizing matrix, eigenvalues (eigenl) + eigenvectors (U_neg)
    U = linalg.inv(U_neg)  # Change of basis matrix
    return eigen_l, U_neg, U


def Integrate(B_eigen, P0, times):  # Eq (9)
    # Given an initial probability vector, P0, return the probability vectors Pt at specified times, given a linear
    # rate matrix dP / dt = B P. Note that the integration process requires computing the eigendecomposition of the
    # matrix to avoid repeating this each call of Integrate. The change of basis matrix, B_eigen is provided, which can
    # be calculated with the GetEigenElements() function

    eigen_l, U_neg, U = B_eigen
    a = np.dot(U, P0)  # Decomposition of the initial condition in the eigenbase
    # For each time point (row), calculate Pt from the exponential components of the eigenvector (eq. 9)
    Pt = np.real(np.dot(a * np.exp(np.tensordot(times, eigen_l, 0)), U_neg.T))
    return np.abs(Pt + errLog)  # To avoid spurious negative probabilities

########################################################################################################################


def GetTransitionMatrix(B_eigen, DeltaT):  # Text after eq (11)
    # Given the transition matrix, B and a step Delta T, return a matrix Q of the probability of transition between
    # states. Each index of i, j of the matrix will contain the probability of transition from a state i to a state j
    # in an interval Delta_t

    eigen_l, U_neg, U = B_eigen
    Q = np.zeros_like(U)  # U and Q have the same size
    Qdim = len(eigen_l)
    for inn in range(Qdim):
        v_aux = np.zeros(Qdim)
        v_aux[inn] = 1  # To get each element, we need to propagate from each possible initial state
        Q[inn, :] = Integrate(B_eigen, v_aux, [DeltaT])
    return Q, np.log(np.abs(Q) + errLog)  # logQ is also returned for efficiency of calculation later on (it is
    # easier to sum then to multiply)


def GetTransitionMatrix_Fast(B, DeltaT, N=13):  # Text after eq (11)
    # Given the transition matrix, B and a step Delta T, return a matrix Q of the probability of transition between
    # states. Each index of i, j of the matrix will contain the probability of transition from a state i to a state j
    # in an interval Delta_t
    # This version is faster than GetTransitionMatrix, but is less accurate by integrating the ME at short times:
    # P(t + Delta_t) = P(t) + A P(t) Delta_t
    err = 1E-12  # To ignore zeros in the log
    Q = np.eye(N + 1) + B * DeltaT + err
    return Q, np.log(Q + errLog)  # logQ is also returned for efficiency of calculation later on (it is easier to sum
    # then to multiply)


def GetTransitionMatrix_Fast2(B, DeltaT, N=13):  # Text after eq (11)
    # Given the transition matrix, B and a step Delta T, return a matrix Q of the probability of transition between
    # states. Each index of i, j of the matrix will contain the probability of transition from a state i to a state j
    # in an interval Delta_t This version is faster than GetTransitionMatrix, but is less accurate by integrating the
    # ME at short times, truncating to second order in the exponent expansion
    # P(t + Delta_t) = P(t) + A P(t) Delta_t + (A * Delta_t)^2 P(t)

    Q2 = np.eye(N + 1) + B * DeltaT / 2
    Q = np.dot(Q2, Q2)
    return Q, np.log(Q + errLog)  # logQ is also returned for efficiency of calculation later on (it is easier to sum
    # then to multiply)


def TransitionMatrix_WithUncertainty(Q, beta):
    P = np.zeros_like(Q)  # The perturbation matrix
    # The resulting transition matrix will be P^T Q P. There are different choices for P. For simplicity, we will use
    # a geometric error P(n|n + j)=P(n|n - j) ~ beta**|j|. If beta = (0,1], 0 is the case without perturbation and 1
    # completely erases the experimental information
    for irow, row in enumerate(P):
        for icol, column in enumerate(row):
            P[irow, icol] = beta ** abs(irow - icol)
        P[irow, :] = P[irow, :] / sum(P[irow, :])  # Normalization
    perturbedQ = np.dot(np.dot(P, Q), P.T)  # P^T Q P
    logPerturbedQ = np.log(perturbedQ + errLog)
    return perturbedQ, logPerturbedQ

########################################################################################################################


def LogLikelihood(LogQ, traj, ignorezeros=True):  # Eq (11)
    # Given a trajectory traj as an array of occupancies in time [n(t0), n(t1), n(t2), ....., n(t_final)] and a
    # transition matrix Q, return the Likelihood of that trajectory
    # ignorezeros = True ignores transitions that include zero stoichiometry
    if ignorezeros:
        LogQ[0, :] = 0
        LogQ[:, 0] = 0
    probs = LogQ[traj[:-1], traj[1:]]  # Selecting the pairs of transitions stored in traj from the vector of
    # Transitions. We will be ignoring P0 since we don't have knowledge a priori on what happens there
    return np.sum(probs)  # Sum of logs is the same as log of product


def LogLikelihood_occurences(LogQ, ocurrences, ignorezeros=True):  # Eq (11)
    # Given a matrix of counters of observations of each possible transition "ocurrences" and a transition matrix Q,
    # return the Likelihood of that set of occurrences
    # ignorezeros = True ignores transitions that include zero stoichiometry
    if ignorezeros:
        ocurrences[0, :] = 0
        ocurrences[:, 0] = 0
    return np.sum(ocurrences * LogQ)  # Sum of logs is the same as log of product
