"""Run the full experiment for several seeds in parallel; results cached in results/seed<k>.pkl."""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import hybrid_rec as H
from joblib import Parallel, delayed

SEEDS = [0, 1, 2]
KS = [5, 10, 20, 40]

def _one(seed):
    f = f"results/seed{seed}.pkl"
    if os.path.exists(f):
        return pickle.load(open(f, "rb"))
    R, M, T = H.load_data()
    nU, nI = R.u.max() + 1, len(M)
    E = np.load("results/emb.npy")
    Cs = {k: H.knn_graph(E, k) for k in KS}
    out = H.run_seed(seed, R, nU, nI, E, Cs, log=lambda m: print(m, flush=True))
    pickle.dump(out, open(f, "wb"))
    return out

def run_all(seeds=SEEDS):
    return Parallel(n_jobs=len(seeds))(delayed(_one)(s) for s in seeds)

if __name__ == "__main__":
    run_all()
