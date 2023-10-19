import torch
import numpy as np
import scipy as sp
import argparse
import pickle
from scipy.stats import ortho_group
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
import json



def gen_hmm_state_data(num_latent_states, sequence_length, state_stickiness=0.999, random_seed=0):
    """Generate a sequence of HMM states.

    Args:
        num_latent_states (int): number of HMM latent states.
        sequence_length (int): number of time steps in the HMM.
        state_stickiness (float): probability of staying in current state (default=0.999).
        random_seed (int): for reproducible stochasticity (default=0).

    Returns:
        (sequence_length,) array of HMM states.
    """
    # set random seed
    rng = np.random.default_rng(random_seed)

    # create transition matrix
    transition_matrix = np.zeros((num_latent_states,
                                  num_latent_states))
    for i in range(num_latent_states):
        for j in range(num_latent_states):
            if j == i:
                transition_matrix[i, j] = state_stickiness
            if j == i+1:
                transition_matrix[i, j] = 1.-state_stickiness
            if i == num_latent_states-1 and j == 0:
                transition_matrix[i, j] = 1.-state_stickiness
    transition_matrix /= transition_matrix.sum(1, keepdims=True)

    # get initial state distrib as left eig vec
    e_vals, e_vecs = sp.linalg.eig(transition_matrix, left=True, right=False)
    one_eig_idx = np.argmin(np.abs(np.real(e_vals)-1.))
    assert np.imag(e_vals[one_eig_idx]) == 0
    ev_one = np.real(e_vecs[:, one_eig_idx])
    init_state_probs = ev_one / ev_one.sum()

    # create latent state sequence
    state_sequence = np.zeros(sequence_length, dtype=int)
    for i in range(sequence_length):
        if i == 0:
            m_draw = rng.multinomial(1, pvals=init_state_probs)
        else:
            m_draw = rng.multinomial(
                1, pvals=transition_matrix[state_sequence[i-1], :])
        state_sequence[i] = np.argmax(m_draw)
    return state_sequence, transition_matrix, init_state_probs


def gen_batch_hmm_state_data(num_latent_states, batch_size, sequence_length,
                             state_stickiness=0.999, random_seed=0):
    # set random seed
    rng = np.random.default_rng(random_seed)

    # create transition matrix
    transition_matrix = np.zeros((num_latent_states,
                                  num_latent_states))
    for i in range(num_latent_states):
        for j in range(num_latent_states):
            if j == i:
                transition_matrix[i, j] = state_stickiness
            if j == i+1:
                transition_matrix[i, j] = 1.-state_stickiness
            if i == num_latent_states-1 and j == 0:
                transition_matrix[i, j] = 1.-state_stickiness
    transition_matrix /= transition_matrix.sum(1, keepdims=True)
    # get initial state distrib as left eig vec
    e_vals, e_vecs = sp.linalg.eig(transition_matrix, left=True, right=False)
    one_eig_idx = np.argmin(np.abs(np.real(e_vals)-1.))
    assert np.imag(e_vals[one_eig_idx]) == 0
    ev_one = np.real(e_vecs[:, one_eig_idx])
    init_state_probs = ev_one / ev_one.sum()

    # create latent state sequence
    state_sequence = np.zeros((batch_size, sequence_length), dtype=int)
    for b in range(batch_size):
        for i in range(sequence_length):
            if i == 0:
                m_draw = rng.multinomial(1, pvals=init_state_probs)
            else:
                m_draw = rng.multinomial(
                    1, pvals=transition_matrix[state_sequence[b,i-1], :])
            state_sequence[b, i] = np.argmax(m_draw)

    return state_sequence, transition_matrix, init_state_probs



def leaky_ReLU_1d(d, negSlope):
    if d > 0:
        return d
    else:
        return d * negSlope
leaky1d = np.vectorize(leaky_ReLU_1d)
def leaky_ReLU(D, negSlope):
    assert negSlope > 0
    return leaky1d(D, negSlope)
def generateUniformMat(Ncomp, condT):
    """
    generate a random matrix by sampling each element uniformly at random
    check condition number versus a condition threshold
    """
    A = np.random.uniform(0, 2, (Ncomp, Ncomp)) - 1
    for i in range(Ncomp):
        A[:, i] /= np.sqrt((A[:, i] ** 2).sum())

    while np.linalg.cond(A) > condT:
        # generate a new A matrix!
        A = np.random.uniform(0, 2, (Ncomp, Ncomp)) - 1
        for i in range(Ncomp):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())

    return A
def pnl_change_gaussian_ts(NClass=5):
    lags = 1
    Nlayer = 3
    length = 3
    condList = []
    negSlope = 0.2
    latent_size = 8
    transitions = []
    noise_scale = 0.1
    batch_size = 40_000
    Niter4condThresh = 1e4

    for i in range(int(Niter4condThresh)):
        # A = np.random.uniform(0,1, (Ncomp, Ncomp))
        A = np.random.uniform(1, 2, (latent_size, latent_size))  # - 1
        for i in range(latent_size):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))

    condThresh = np.percentile(condList, 25)  # only accept those below 25% percentile
    for l in range(lags):
        B = generateUniformMat(latent_size, condThresh)
        transitions.append(B)
    transitions.reverse()

    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        A = ortho_group.rvs(latent_size)  # generateUniformMat(Ncomp, condThresh)
        mixingList.append(A)
    # Domain-varying edges
    edge_pairs = [(1,2), (3,4)]
    edge_weights = np.random.uniform(-1,1,(NClass, len(edge_pairs)))
    yt = []; xt = []; ct = []
    yt_ns = []; xt_ns = []; ct_ns = []

    for j in range(NClass):
        ct.append(j * np.ones(batch_size))
        y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
        y_l = (y_l - np.mean(y_l, axis=0 ,keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)
        for p_idx, pair in enumerate(edge_pairs):
            transitions[0][pair[0], pair[1]] = edge_weights[j, p_idx]
        for i in range(lags):
            yt.append(y_l[:,i,:])
        mixedDat = np.copy(y_l)
        for l in range(Nlayer - 1):
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            mixedDat = np.dot(mixedDat, mixingList[l])
        x_l = np.copy(mixedDat)
        for i in range(lags):
            xt.append(x_l[:,i,:])
            
        # Mixing function
        for i in range(length):
            # Transition function
            y_t = np.random.normal(0, noise_scale, (batch_size, latent_size))
            for l in range(lags):
                y_t += leaky_ReLU(np.dot(y_l[:,l,:], transitions[l]), negSlope)
            y_t = leaky_ReLU(y_t, negSlope)
            yt.append(y_t)
            # Mixing function
            mixedDat = np.copy(y_t)
            for l in range(Nlayer - 1):
                mixedDat = leaky_ReLU(mixedDat, negSlope)
                mixedDat = np.dot(mixedDat, mixingList[l])
            x_t = np.copy(mixedDat)
            xt.append(x_t)
            y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]

        yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2); ct = np.array(ct).transpose(1,0)
        yt_ns.append(yt); xt_ns.append(xt); ct_ns.append(ct)
        yt = []; xt = []; ct = []                

    yt_ns = np.vstack(yt_ns)
    xt_ns = np.vstack(xt_ns)
    ct_ns = np.vstack(ct_ns)
    ct_ns = np.broadcast_to(ct_ns, (yt_ns.shape[:2]))
    return {
        "z": yt_ns,
        "x": xt_ns,
        "c": ct_ns,
    }

def arhmm_pnl_change_gaussian_ts(NClass=5, lags=1, Nlayer=3, length=3, negSlope=0.2, latent_size=8, noise_scale=0.1, batch_size=40_000, Niter4condThresh=1e4, state_stickiness=0.999, random_seed=0):
    condList = []
    transitions = []

    rng = np.random.default_rng(random_seed)
    
    for i in range(int(Niter4condThresh)):
        # A = np.random.uniform(0,1, (Ncomp, Ncomp))
        A = np.random.uniform(1, 2, (latent_size, latent_size))  # - 1
        for i in range(latent_size):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))

    condThresh = np.percentile(condList, 25)  # only accept those below 25% percentile
    for l in range(lags):
        B = generateUniformMat(latent_size, condThresh)
        transitions.append(B)
    # transitions.reverse()
    transitions_list = [deepcopy(transitions) for _ in range(NClass)]
    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        A = ortho_group.rvs(latent_size)  # generateUniformMat(Ncomp, condThresh)
        mixingList.append(A)
    # Domain-varying edges
    # edge_pairs = [(1,2), (3,4)]
    # edge_weights = np.random.uniform(-1,1,(NClass, len(edge_pairs)))
    
    # for j in range(NClass): 
    #     for p_idx, pair in enumerate(edge_pairs):
    #             transitions_list[j][0][pair[0], pair[1]] = edge_weights[j, p_idx]

    permutations = set()
    P_matrix_list = []
    while len(permutations) < NClass:
        P_l = rng.permutation(latent_size)
        if tuple(P_l) not in permutations:
            permutations.add(tuple(P_l))
            P = np.zeros((latent_size, latent_size))
            P[np.arange(latent_size), P_l] = 1
            P_matrix_list.append(P)
    for j in range(NClass): 
        transitions_list[j][0] = P_matrix_list[j]

    batch_state_sequence, transition_matrix, init_state_probs = gen_batch_hmm_state_data(
        num_latent_states=NClass, batch_size=batch_size, sequence_length=length+lags, state_stickiness=state_stickiness, random_seed=rng)
    
    batch_Z = np.zeros((batch_size, length+lags, latent_size))
    batch_X = np.zeros((batch_size, length+lags, latent_size))
    batch_C = batch_state_sequence # (batch_size, sequence_length)
    
    
    # (batch_size, lags, latent_size)
    batch_z_lags = np.random.normal(0, 1, (batch_size, lags, latent_size))
    batch_z_lags = (batch_z_lags - np.mean(batch_z_lags, axis=0 ,keepdims=True)) / np.std(batch_z_lags, axis=0 ,keepdims=True)
    
    batch_Z[:, :lags, :] = np.copy(batch_z_lags)
    
    mixedDat = np.copy(batch_z_lags)
    for l in range(Nlayer - 1):
        mixedDat = leaky_ReLU(mixedDat, negSlope)
        mixedDat = np.dot(mixedDat, mixingList[l])
    batch_X[:,:lags,:] = np.copy(mixedDat) # (batch_size, lags, latent_size)
    
    for b in tqdm(range(batch_size)):
        for t in range(length):
            c_t = batch_state_sequence[b,t]
            z_t = np.random.normal(0, noise_scale, (latent_size,))
            for l in range(lags):
                z_t += leaky_ReLU(np.dot(batch_Z[b,t+lags-l-1,:], transitions_list[c_t][l]), negSlope)
            z_t = leaky_ReLU(z_t, negSlope)
            batch_Z[b,t+lags,:] = np.copy(z_t)
            # Mixing function
            mixedDat = np.copy(z_t)
            for l in range(Nlayer - 1):
                mixedDat = leaky_ReLU(mixedDat, negSlope)
                mixedDat = np.dot(mixedDat, mixingList[l])
            x_t = np.copy(mixedDat)
            batch_X[b,t+lags,:] = np.copy(x_t)
    return {
        "data":{
            "Z": batch_Z,
            "X": batch_X,
            "C": batch_C,
            "A": transition_matrix
        },
        "meta": {
            "NClass": NClass,
            "lags": lags,
            "Nlayer": Nlayer,
            "length": length,
            "negSlope":negSlope,
            "latent_size": latent_size,
            "noise_scale": noise_scale,
            "batch_size": batch_size,
            "Niter4condThresh": Niter4condThresh,
            "state_stickiness": state_stickiness,
            "random_seed": random_seed
        }
    }

def write_arhmm_pnl_change_gaussian_ts(NClass=5, lags=1, Nlayer=3, length=3, negSlope=0.2, latent_size=8, noise_scale=0.1, batch_size=40_000, Niter4condThresh=1e4, state_stickiness=0.999, random_seed=0):
    data = arhmm_pnl_change_gaussian_ts(NClass=NClass, lags=lags, Nlayer=Nlayer, length=length, negSlope=negSlope, latent_size=latent_size, noise_scale=noise_scale, batch_size=batch_size, Niter4condThresh=Niter4condThresh, state_stickiness=state_stickiness, random_seed=random_seed)
    data_path = Path(f"../data/simulation/z{latent_size}_c{NClass}_lags{lags}_len{length}_Nlayer{Nlayer}/arhmm_pnl_change_gaussian_ts")
    data_path.mkdir(parents=True, exist_ok=True)
    with open(data_path/"data.pkl", "wb") as f:
        pickle.dump(data["data"], f)
    with open(data_path/"meta.json", "w") as f:
        json.dump(data["meta"], f,indent=2)
if __name__ == "__main__":
    length = 4
    lags=2
    Nlayer=3
    write_arhmm_pnl_change_gaussian_ts(
        batch_size = 200_000,
        state_stickiness=0.9,
        lags=lags,
        length=length,
        latent_size=8,
        noise_scale=0.1,
        NClass=5,
        Nlayer=Nlayer,
        negSlope=0.2,
        random_seed=42)
    
    