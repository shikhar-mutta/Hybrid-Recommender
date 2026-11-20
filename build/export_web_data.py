"""Export everything the web interface needs to web/public/data/*.json  (run from the project root)."""
import os, sys, json, pickle
sys.path.insert(0, "src")
import numpy as np, pandas as pd, scipy.sparse as sp
import hybrid_rec as H

OUT = "web/public/data"
os.makedirs(OUT, exist_ok=True)
dump = lambda name, obj: json.dump(obj, open(f"{OUT}/{name}.json", "w"), separators=(",", ":"), default=float)

R, M, T = H.load_data()
nU, nI = int(R.u.max()) + 1, len(M)
E = np.load("results/emb.npy")
res = [pickle.load(open(f"results/seed{s}.pkl", "rb")) for s in range(3)]
P0 = res[0]["params"]
C = H.knn_graph(E, P0["k"])
ctx = H.Ctx(R, R.iloc[:0], nU, nI)
title = M.title.values

# ---------------------------------------------------------------- aggregated results
def ms(getter):
    v = np.array([getter(r) for r in res], float)
    return [round(float(v.mean()), 4), round(float(v.std()), 4)]

results = {
    "params": P0,
    "main": {n: {m: ms(lambda r: r["main"][n][m]) for m in ["p", "r", "ndcg", "tail", "coverage", "mean_pop"]} for n in res[0]["main"]},
    "lam_sweep": {str(l): {m: ms(lambda r: r["lam_sweep"][l][m]) for m in ["p", "r", "ndcg", "tail"]} for l in H.LAMS},
    "cold_sweep": {str(l): {m: ms(lambda r: r["cold_sweep"][l][m]) for m in ["p", "r", "user_lift"]} for l in H.LAMS},
    "cold_main": {n: {m: ms(lambda r: r["cold_main"][n][m]) for m in ["p", "r", "ndcg", "user_p", "user_lift"]} for n in res[0]["cold_main"]},
    "k_sweep": {str(k): {m: ms(lambda r: r["k_sweep"][k][m]) for m in ["p", "r"]} for k in res[0]["k_sweep"]},
    "c_sweep": {str(c): {m: ms(lambda r: r["c_sweep"][c][m]) for m in ["p", "r"]} for c in res[0]["c_sweep"]},
    "n_cold": res[0]["n_cold"],
    "dataset": {"users": nU, "films": nI, "ratings": len(R), "tags": len(T),
                "density": round(len(R) / (nU * nI), 4)},
}
dump("results", results)

# ---------------------------------------------------------------- models on all ratings
print("scoring...", flush=True)
nrm = np.sqrt(np.asarray(ctx.A.multiply(ctx.A).sum(0)).ravel()); nrm[nrm == 0] = 1
An = ctx.A @ sp.diags(1 / nrm)
Sitem = (An.T @ An).tocsr(); Sitem.setdiag(0); Sitem.eliminate_zeros()
methods = {
    "Item-based CF": H.item_cf_scores(ctx),
    "RWR (ratings only)": H.rwr_scores(ctx, C, 0.0, P0["c"]),
    "Content only": H.content_profile_scores(ctx, E),
    "Hybrid RWR": H.rwr_scores(ctx, C, P0["lam"], P0["c"]),
    "Hybrid + popularity discount": H.debias(H.rwr_scores(ctx, C, P0["lam_debias"], P0["c"]), ctx.pop, P0["beta_hyb"]),
    "node2vec": H.node2vec_scores(H.build_P(ctx.A, C, P0["lam"]), nU),
}
tops = {n: H.top_k(S, ctx, 10) for n, S in methods.items()}

# ---------------------------------------------------------------- communities
W = H.user_projection(R, nU, nI, k=15)
lab, Q = H.louvain(W, seed=0)
genres = sorted({g for gs in M.genres for g in gs.split("|") if g != "(no genres listed)"})
GP = H.genre_profiles(R, M, nU, genres)
glob = GP.mean(0)
sizes = np.bincount(lab)
spread = lambda L: sum((L == c).sum() * ((GP[L == c].mean(0) - glob) ** 2).sum() for c in np.unique(L))
rng = np.random.default_rng(0)
obs = spread(lab); null = np.array([spread(rng.permutation(lab)) for _ in range(2000)])
big = [c for c in range(len(sizes)) if sizes[c] >= 20]
dump("communities", {
    "genres": genres, "modularity": round(float(Q), 3), "sizes": sizes.tolist(),
    "spread": round(float(obs), 3), "null_mean": round(float(null.mean()), 3), "null_sd": round(float(null.std()), 3),
    "p": round(float((1 + (null >= obs).sum()) / 2001), 4),
    "communities": [{"id": int(c), "size": int(sizes[c]), "lift": [round(float(x), 3) for x in GP[lab == c].mean(0) / np.maximum(glob, 1e-9)]} for c in big],
})

# ---------------------------------------------------------------- films, users
print("films/users...", flush=True)
cn = []
for i in range(nI):
    row = C[i]
    o = row.indices[np.argsort(-row.data)][:6]
    ro = Sitem[i]
    oc = ro.indices[np.argsort(-ro.data)][:6]
    cn.append({"c": [int(x) for x in o], "r": [int(x) for x in oc]})
dump("films", {"title": title.tolist(), "genres": M.genres.tolist(), "pop": ctx.pop.tolist(), "nb": cn})
users = []
for u in range(nU):
    L = ctx.Liked[u]
    fav = L.indices[np.argsort(-L.data, kind="stable")][:12]
    users.append({"id": u + 1, "n_liked": int(L.nnz), "n_rated": int((ctx.A[u] > 0).sum()), "community": int(lab[u]),
                  "fav": [int(x) for x in fav], "recs": {n: [int(x) for x in t[u]] for n, t in tops.items()}})
dump("users", users)
print("done")
