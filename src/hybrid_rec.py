"""
Hybrid graph recommender (AIM608 Project 4).

Everything the notebook reports is computed by the functions in this file.

Graph
-----
Nodes  : users (0..nU-1) then films (nU..nU+nI-1).
Edges  : rating edges (user-film, weight = rating) and
         content edges (film-film, k nearest neighbours in sentence-embedding space).
Walk   : random walk with restart (personalised PageRank).  A film node follows a
         content edge with probability `lam` and a rating edge with probability
         `1-lam`; a film with no ratings (cold) always follows content edges.
"""
import os
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp

LIKE = 4.0          # a rating >= LIKE counts as "liked"
DATA = "data/ml-latest-small"


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_data(path=DATA):
    """Return ratings R (cols u,i,rating), movies M (index = film id i), tags T."""
    R = pd.read_csv(f"{path}/ratings.csv")
    M = pd.read_csv(f"{path}/movies.csv").reset_index(drop=True)
    T = pd.read_csv(f"{path}/tags.csv")
    M["i"] = M.index
    R["u"] = R.userId.astype("category").cat.codes.astype(int)
    R["i"] = R.movieId.map(dict(zip(M.movieId, M.i)))
    return R, M, T


def film_text(M, T):
    """One text per film: title, genres and user tags (used for the embedding)."""
    tags = T.assign(tag=T.tag.astype(str).str.lower()).groupby("movieId").tag.apply(
        lambda s: ", ".join(sorted(set(s))))
    title = M.title.str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
    genres = M.genres.str.replace("(no genres listed)", "unknown", regex=False).str.replace("|", ", ", regex=False)
    tg = M.movieId.map(tags).fillna("")
    return (title + ". Genres: " + genres + ". Tags: " + tg).str.strip().tolist()


def embed_texts(texts, model="all-MiniLM-L6-v2", cache=None):
    if cache and os.path.exists(cache):
        return np.load(cache)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model)
    E = m.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    if cache:
        np.save(cache, E)
    return E


# ----------------------------------------------------------------------------
# splits
# ----------------------------------------------------------------------------
def holdout_mask(R, frac, rng):
    """Boolean mask hiding `frac` of EACH user's ratings (at least one)."""
    key = pd.Series(rng.random(len(R)), index=R.index)
    rank = key.groupby(R.u).rank(method="first")
    n = R.groupby("u").u.transform("size")
    return (rank <= np.maximum(1, np.round(frac * n))).values


def cold_films(R, rng, frac=0.10, min_ratings=5):
    """Choose films whose ratings are ALL hidden (cold start)."""
    cnt = R.groupby("i").size()
    elig = cnt[cnt >= min_ratings].index.values
    return np.sort(rng.choice(elig, int(round(frac * len(elig))), replace=False))


# ----------------------------------------------------------------------------
# matrices
# ----------------------------------------------------------------------------
class Ctx:
    """Everything derived from one (train, test) pair.  Built from TRAIN only."""

    def __init__(self, train, test, nU, nI):
        self.nU, self.nI = nU, nI
        self.A = sp.csr_matrix((train.rating.values.astype(np.float32), (train.u, train.i)), shape=(nU, nI))
        lk = train[train.rating >= LIKE]
        self.Liked = sp.csr_matrix((lk.rating.values.astype(np.float32), (lk.u, lk.i)), shape=(nU, nI))
        s = self.A.tocoo()
        self.seen_r, self.seen_c = s.row, s.col
        self.pop = np.asarray((self.A > 0).sum(0)).ravel()            # ratings per film (train)
        nrated = int((self.pop > 0).sum())
        order = np.argsort(-self.pop, kind="stable")
        self.top10 = np.zeros(nI, bool)
        self.top10[order[: int(np.ceil(0.10 * nrated))]] = True        # top 10% most rated films
        self.rel = np.zeros((nU, nI), bool)
        t = test[test.rating >= LIKE]
        self.rel[t.u.values, t.i.values] = True


def rownorm(X):
    X = sp.csr_matrix(X, dtype=np.float64)
    d = np.asarray(X.sum(1)).ravel()
    d[d == 0] = 1.0
    return sp.diags(1.0 / d) @ X


# ----------------------------------------------------------------------------
# content graph
# ----------------------------------------------------------------------------
def knn_graph(E, k):
    """Symmetric k-nearest-neighbour film graph; weight = cosine similarity."""
    n = len(E)
    S = E @ E.T
    np.fill_diagonal(S, -1)
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    vals = np.clip(S[np.arange(n)[:, None], idx].ravel(), 1e-6, None)
    C = sp.csr_matrix((vals, (rows, idx.ravel())), shape=(n, n))
    return C.maximum(C.T).tocsr()


# ----------------------------------------------------------------------------
# recommenders (all return a dense nU x nI score matrix)
# ----------------------------------------------------------------------------
def popularity_scores(ctx):
    return np.tile(ctx.pop.astype(np.float32), (ctx.nU, 1))


def item_cf_scores(ctx):
    """Item-based CF: project user-film graph onto films (cosine of rating columns)."""
    A = ctx.A
    nrm = np.sqrt(np.asarray(A.multiply(A).sum(0)).ravel())
    nrm[nrm == 0] = 1
    An = A @ sp.diags(1 / nrm)
    S = (An.T @ An).tocsr()
    S.setdiag(0)
    S.eliminate_zeros()
    return (ctx.Liked @ S).toarray().astype(np.float32)


def content_profile_scores(ctx, E):
    """Content-based: cosine between film embedding and the user's liked-film profile."""
    prof = ctx.Liked @ E
    nrm = np.linalg.norm(prof, axis=1, keepdims=True)
    nrm[nrm == 0] = 1
    return ((prof / nrm) @ E.T).astype(np.float32)


def build_P(A, C, lam):
    """Column-stochastic transition matrix of the hybrid graph (users first, then films)."""
    nU, nI = A.shape
    has_r = np.asarray((A > 0).sum(0)).ravel() > 0
    has_c = np.asarray(C.sum(1)).ravel() > 0
    lamv = np.where(has_r, lam, 1.0)
    lamv = np.where(has_c, lamv, 0.0)
    P_FU = rownorm(A).T                                   # user -> film
    P_UF = rownorm(A.T.tocsr()).T @ sp.diags(1 - lamv)    # film -> user  (prob 1-lam)
    P_FF = rownorm(C).T @ sp.diags(lamv)                  # film -> film  (prob lam)
    P = sp.bmat([[None, P_UF], [P_FU, P_FF]], format="csr")
    return P.astype(np.float32)


def rwr_all(P, nU, c=0.5, tol=1e-5, maxit=120, users=None):
    """Personalised PageRank from every user by power iteration:  r <- (1-c) P r + c e_u.
    Returns the film part of the stationary vectors, shape (len(users), nI)."""
    N = P.shape[0]
    users = np.arange(nU) if users is None else np.asarray(users)
    E = np.zeros((N, len(users)), np.float32)
    E[users, np.arange(len(users))] = 1
    r = E.copy()
    for _ in range(maxit):
        rn = (1 - c) * (P @ r) + c * E
        delta = np.abs(rn - r).sum(0).max()
        r = rn
        if delta < tol:
            break
    return r[nU:].T.copy()


def rwr_scores(ctx, C, lam, c):
    return rwr_all(build_P(ctx.A, C, lam), ctx.nU, c)


def debias(S, pop, beta):
    """Popularity discount: divide scores by (1+ratings)^beta."""
    return S / (1.0 + pop[None, :].astype(np.float32)) ** beta


def minmax_rows(S):
    lo, hi = S.min(1, keepdims=True), S.max(1, keepdims=True)
    return (S - lo) / np.where(hi - lo == 0, 1, hi - lo)


def late_fusion(S_cf, S_content, alpha):
    """Score-level blend (a non-graph hybrid used as a comparison)."""
    return (1 - alpha) * minmax_rows(S_cf) + alpha * minmax_rows(S_content)


# ----------------------------------------------------------------------------
# node2vec / DeepWalk embeddings (own implementation, torch skip-gram)
# ----------------------------------------------------------------------------
def _walks(P, n_walks, length, rng):
    Q = P.T.tocsr().astype(np.float64)                    # row j = out-probabilities of node j
    N = Q.shape[0]
    cum = np.cumsum(Q.data)
    before = np.concatenate([[0.0], cum])[Q.indptr[:-1]]
    rowid = np.repeat(np.arange(N), np.diff(Q.indptr))
    cs = cum - np.repeat(before, np.diff(Q.indptr)) + rowid
    cur = np.tile(np.arange(N), n_walks)
    out = np.empty((len(cur), length), np.int32)
    out[:, 0] = cur
    for t in range(1, length):
        q = cur + rng.random(len(cur)) * 0.999999
        pos = np.searchsorted(cs, q)
        pos = np.minimum(pos, Q.indptr[cur + 1] - 1)
        cur = Q.indices[pos]
        out[:, t] = cur
    return out


def node2vec_embed(P, dim=64, n_walks=5, length=30, window=4, neg=5, epochs=2,
                   batch=16384, lr=0.01, seed=0, threads=2):
    """Skip-gram with negative sampling on random walks over the hybrid graph.
    Walks are first-order (node2vec with p = q = 1, i.e. DeepWalk)."""
    import torch
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    N = P.shape[0]
    W = _walks(P, n_walks, length, rng)
    freq = np.bincount(W.ravel(), minlength=N).astype(np.float64) ** 0.75
    freq = torch.tensor(freq / freq.sum(), dtype=torch.float32)
    src, dst = [], []
    for o in range(1, window + 1):
        a, b = W[:, :-o].ravel(), W[:, o:].ravel()
        src += [a, b]
        dst += [b, a]
    src = torch.from_numpy(np.concatenate(src)).long()
    dst = torch.from_numpy(np.concatenate(dst)).long()
    emb_in = torch.nn.Embedding(N, dim, sparse=True)
    emb_out = torch.nn.Embedding(N, dim, sparse=True)
    torch.nn.init.uniform_(emb_in.weight, -0.5 / dim, 0.5 / dim)
    torch.nn.init.zeros_(emb_out.weight)
    opt = torch.optim.SparseAdam(list(emb_in.parameters()) + list(emb_out.parameters()), lr=lr)
    for ep in range(epochs):
        perm = torch.randperm(len(src))
        for s in range(0, len(src), batch):
            ix = perm[s:s + batch]
            u, v = emb_in(src[ix]), emb_out(dst[ix])
            nv = emb_out(torch.multinomial(freq, len(ix) * neg, replacement=True).view(len(ix), neg))
            pos = torch.nn.functional.logsigmoid((u * v).sum(1))
            ng = torch.nn.functional.logsigmoid(-(nv @ u.unsqueeze(2)).squeeze(2)).sum(1)
            loss = -(pos + ng).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return emb_in.weight.detach().numpy()


def node2vec_scores(P, nU, **kw):
    Z = node2vec_embed(P, **kw)
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    return (Z[:nU] @ Z[nU:].T).astype(np.float32)


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------
def top_k(S, ctx, k=10, cand=None, rng=None):
    """Top-k unseen films per user (ties between equal scores broken at random)."""
    S = S.astype(np.float32).copy()
    rng = rng or np.random.default_rng(0)
    z = S == 0
    S[z] = rng.random(int(z.sum()), dtype=np.float32) * 1e-30
    S[ctx.seen_r, ctx.seen_c] = -np.inf
    if cand is not None:
        S[:, ~cand] = -np.inf
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    o = np.argsort(-np.take_along_axis(S, idx, 1), axis=1)
    return np.take_along_axis(idx, o, 1)


def evaluate(S, ctx, k=10, cand=None, seed=0):
    """precision@k, recall@k, NDCG@k + popularity statistics, averaged over users
    that have at least one relevant (hidden, liked) film."""
    top = top_k(S, ctx, k, cand, np.random.default_rng(seed))
    users = np.where(ctx.rel.sum(1) > 0)[0]
    t = top[users]
    hits = ctx.rel[users[:, None], t]
    nrel = ctx.rel[users].sum(1)
    disc = 1 / np.log2(np.arange(k) + 2)
    dcg = (hits * disc).sum(1)
    idcg = np.array([disc[: min(n, k)].sum() for n in nrel])
    return dict(
        p=hits.mean(1).mean(), r=(hits.sum(1) / nrel).mean(), ndcg=(dcg / idcg).mean(),
        tail=(~ctx.top10[t]).mean(),                       # share outside top-10% popular films
        coverage=len(np.unique(t)) / (cand.sum() if cand is not None else ctx.nI),
        mean_pop=np.log1p(ctx.pop[t]).mean(),
        n_users=len(users))


def evaluate_users_for_cold_films(S, ctx, cand, k=10, seed=0):
    """Reverse direction: for each cold film, rank users (by the film's percentile rank in
    each user's own cold list) and measure precision@k over the users that liked it."""
    rng = np.random.default_rng(seed)
    cols = np.where(cand)[0]
    Sc = S[:, cols].astype(np.float32).copy()
    z = Sc == 0
    Sc[z] = rng.random(int(z.sum()), dtype=np.float32) * 1e-30
    rank = np.argsort(np.argsort(-Sc, axis=1), axis=1)          # per-user rank of each cold film
    rel = ctx.rel[:, cols]
    prec, lift = [], []
    for j in range(len(cols)):
        n = rel[:, j].sum()
        if n == 0:
            continue
        key = rank[:, j] + rng.random(ctx.nU) * 1e-3
        top = np.argsort(key)[:k]
        p = rel[top, j].mean()
        prec.append(p)
        lift.append(p / (n / ctx.nU))
    return dict(user_p=float(np.mean(prec)), user_lift=float(np.mean(lift)), n_films=len(prec))


# ----------------------------------------------------------------------------
# full experiment for one random seed
# ----------------------------------------------------------------------------
LAMS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]   # weight of content edges
BETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75]                            # popularity-discount exponents


def run_seed(seed, R, nU, nI, E, Cs, k_main=20, n2v=True, log=print):
    t0 = time.time()
    C = Cs[k_main]
    rng = np.random.default_rng(seed)
    out = {"seed": seed}

    # ---- split: test (20%/user), then validation inside train (15%/user) -------------
    te = holdout_mask(R, 0.20, rng)
    trainval, test = R[~te], R[te]
    va = holdout_mask(trainval, 0.15, rng)
    ctx_v = Ctx(trainval[~va], trainval[va], nU, nI)       # hyper-parameter tuning only
    ctx_t = Ctx(trainval, test, nU, nI)                     # final numbers

    # ---- tune on validation -------------------------------------------------------
    val = {}
    best_c, best_p = None, -1
    for c in [0.3, 0.5, 0.7]:
        m = evaluate(rwr_scores(ctx_v, C, 0.0, c), ctx_v)
        val[f"rwr_c={c}"] = m["p"]
        if m["p"] > best_p:
            best_c, best_p = c, m["p"]
    Sv_cf = rwr_scores(ctx_v, C, 0.0, best_c)
    beta_cf = max(BETAS, key=lambda b: evaluate(debias(Sv_cf, ctx_v.pop, b), ctx_v)["p"])
    best_lam, best_p = None, -1
    best_pair, best_pp = None, -1
    for lam in [0.05, 0.1, 0.15, 0.2, 0.3]:
        Sv = rwr_scores(ctx_v, C, lam, best_c)
        for b in BETAS:
            pp = evaluate(debias(Sv, ctx_v.pop, b), ctx_v)["p"]
            val[f"hyb_lam={lam}_beta={b}"] = pp
            if b == 0 and pp > best_p:
                best_lam, best_p = lam, pp
            if pp > best_pp:
                best_pair, best_pp = (lam, b), pp
    lam_d, beta_h = best_pair
    Sv_icf, Sv_ct = item_cf_scores(ctx_v), content_profile_scores(ctx_v, E)
    al = {a: evaluate(late_fusion(Sv_icf, Sv_ct, a), ctx_v)["p"] for a in [.02, .05, .1, .2, .4]}
    alpha = max(al, key=al.get)
    out["params"] = dict(c=best_c, lam=best_lam, lam_debias=lam_d, beta_cf=beta_cf, beta_hyb=beta_h, alpha=alpha, k=k_main)
    out["val"] = val
    log(f"[seed {seed}] tuned {out['params']}  ({time.time() - t0:.0f}s)")

    # ---- test: all methods ---------------------------------------------------------
    sweep = {}
    S_by_lam = {}
    for lam in LAMS:
        S_by_lam[lam] = rwr_scores(ctx_t, C, lam, best_c)
        sweep[lam] = evaluate(S_by_lam[lam], ctx_t)
    out["lam_sweep"] = sweep
    S_cf, S_h = S_by_lam[0.0], S_by_lam[best_lam]
    S_icf, S_ct = item_cf_scores(ctx_t), content_profile_scores(ctx_t, E)
    methods = {
        "Popularity": popularity_scores(ctx_t),
        "Item-based CF": S_icf,
        "Content only (profile)": S_ct,
        "RWR (ratings only)": S_cf,
        "RWR + popularity discount": debias(S_cf, ctx_t.pop, beta_cf),
        "Late fusion (CF+content)": late_fusion(S_icf, S_ct, alpha),
        "Hybrid RWR": S_h,
        "Hybrid RWR + popularity discount": debias(S_by_lam[lam_d], ctx_t.pop, beta_h),
    }
    if n2v:
        methods["node2vec on hybrid graph"] = node2vec_scores(build_P(ctx_t.A, C, best_lam), nU, seed=seed)
    out["main"] = {n: evaluate(S, ctx_t) for n, S in methods.items()}
    log(f"[seed {seed}] test done ({time.time() - t0:.0f}s)")

    # ---- k (content neighbours) sweep ------------------------------------------------
    ksw = {}
    for k, Ck in Cs.items():
        ksw[k] = evaluate(rwr_scores(ctx_t, Ck, best_lam, best_c), ctx_t) if k != k_main else sweep[best_lam]
    out["k_sweep"] = ksw

    # ---- restart-probability sweep (hybrid) ---------------------------------------------
    out["c_sweep"] = {c: (evaluate(rwr_scores(ctx_t, C, best_lam, c), ctx_t) if c != best_c else sweep[best_lam])
                      for c in [0.15, 0.3, 0.5, 0.7, 0.85]}

    # ---- cold start: hide ALL ratings of 10% of the films ---------------------------------
    cold = cold_films(R, rng)
    is_cold = np.zeros(nI, bool)
    is_cold[cold] = True
    m_cold = R.i.isin(cold).values
    ctx_c = Ctx(R[~m_cold], R[m_cold], nU, nI)
    cand = is_cold
    csweep = {}
    S_cold_by_lam = {}
    for lam in LAMS:
        S_cold_by_lam[lam] = rwr_scores(ctx_c, C, lam, best_c)
        e = evaluate(S_cold_by_lam[lam], ctx_c, cand=cand)
        e.update(evaluate_users_for_cold_films(S_cold_by_lam[lam], ctx_c, cand))
        csweep[lam] = e
    out["cold_sweep"] = csweep
    cm = {
        "Random": np.random.default_rng(seed).random((nU, nI)).astype(np.float32),
        "Item-based CF": item_cf_scores(ctx_c),
        "RWR (ratings only)": S_cold_by_lam[0.0],
        "Content only (profile)": content_profile_scores(ctx_c, E),
        "Hybrid RWR": S_cold_by_lam[best_lam],
    }
    if n2v:
        cm["node2vec on hybrid graph"] = node2vec_scores(build_P(ctx_c.A, C, best_lam), nU, seed=seed)
    out["cold_main"] = {}
    for n, S in cm.items():
        e = evaluate(S, ctx_c, cand=cand)
        e.update(evaluate_users_for_cold_films(S, ctx_c, cand))
        out["cold_main"][n] = e
    out["n_cold"] = int(len(cold))
    log(f"[seed {seed}] all done ({time.time() - t0:.0f}s)")
    return out


# ----------------------------------------------------------------------------
# communities of taste
# ----------------------------------------------------------------------------
def user_projection(R, nU, nI, k=15):
    """Project the user-film graph onto users: cosine of mean-centred rating vectors, kNN graph."""
    r = R.copy()
    r["c"] = r.rating - r.groupby("u").rating.transform("mean")
    X = sp.csr_matrix((r.c.values, (r.u, r.i)), shape=(nU, nI))
    nrm = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())
    nrm[nrm == 0] = 1
    X = sp.diags(1 / nrm) @ X
    S = (X @ X.T).toarray()
    np.fill_diagonal(S, -1)
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    rows = np.repeat(np.arange(nU), k)
    vals = np.clip(S[np.arange(nU)[:, None], idx].ravel(), 1e-6, None)
    W = sp.csr_matrix((vals, (rows, idx.ravel())), shape=(nU, nU))
    return W.maximum(W.T).tocsr()


def louvain(W, seed=0, resolution=1.0):
    import networkx as nx
    G = nx.from_scipy_sparse_array(W)
    comms = nx.community.louvain_communities(G, weight="weight", resolution=resolution, seed=seed)
    lab = np.zeros(W.shape[0], int)
    for ci, c in enumerate(sorted(comms, key=len, reverse=True)):
        lab[list(c)] = ci
    return lab, nx.community.modularity(G, [set(np.where(lab == c)[0]) for c in np.unique(lab)], weight="weight")


def genre_profiles(R, M, nU, genres):
    """Per user: share of their LIKED films in each genre."""
    G = np.zeros((len(M), len(genres)), np.float32)
    for gi, g in enumerate(genres):
        G[:, gi] = M.genres.str.contains(g, regex=False).values
    L = R[R.rating >= LIKE]
    X = sp.csr_matrix((np.ones(len(L)), (L.u, L.i)), shape=(nU, len(M)))
    prof = X @ G
    return prof / np.maximum(prof.sum(1, keepdims=True), 1e-9)


# ----------------------------------------------------------------------------
# helpers for the warm-up and the demo
# ----------------------------------------------------------------------------
def power_iteration(P, start, c=0.15, tol=1e-12, maxit=1000):
    """Explicit single-source power iteration; returns the vector and the L1 error history."""
    N = P.shape[0]
    e = np.zeros(N)
    e[start] = 1
    r, hist = e.copy(), []
    P = P.astype(np.float64)
    for _ in range(maxit):
        rn = (1 - c) * (P @ r) + c * e
        hist.append(np.abs(rn - r).sum())
        r = rn
        if hist[-1] < tol:
            break
    return r, hist


def monte_carlo_rwr(P, start, c=0.15, walkers=3000, steps=400, burn=50, seed=0):
    """Simulate the walk with restart; visit frequencies converge to the same vector."""
    rng = np.random.default_rng(seed)
    Q = P.T.tocsr().astype(np.float64)
    N = Q.shape[0]
    cum = np.cumsum(Q.data)
    cnt = np.diff(Q.indptr)
    before = np.concatenate([[0.0], cum])[Q.indptr[:-1]]
    cs = cum - np.repeat(before, cnt) + np.repeat(np.arange(N), cnt)
    cur = np.full(walkers, start)
    visits = np.zeros(N)
    for t in range(steps):
        pos = np.minimum(np.searchsorted(cs, cur + rng.random(walkers) * 0.999999), Q.indptr[cur + 1] - 1)
        cur = Q.indices[pos]
        cur = np.where(rng.random(walkers) < c, start, cur)
        if t >= burn:
            visits += np.bincount(cur, minlength=N)
    return visits / visits.sum()
