"""Sanity tests:  python3 tests/test_project.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, scipy.sparse as sp
import hybrid_rec as H

R, M, T = H.load_data()
nU, nI = int(R.u.max()) + 1, len(M)
E = np.load("results/emb.npy")
C = H.knn_graph(E, 20)
rng = np.random.default_rng(0)
te = H.holdout_mask(R, 0.2, rng)
train, test = R[~te], R[te]
ctx = H.Ctx(train, test, nU, nI)

def test_split_no_leakage():
    key = lambda d: set(zip(d.u, d.i))
    assert not key(train) & key(test)
    assert ctx.A.nnz == len(train)                         # graph has only training ratings
    assert (test.groupby("u").size() >= 1).all() and abs(len(test)/len(R) - 0.2) < 0.01

def test_transition_matrix_stochastic():
    for lam in [0.0, 0.1, 1.0]:
        P = H.build_P(ctx.A, C, lam)
        assert np.allclose(np.asarray(P.sum(0)).ravel(), 1, atol=1e-5), lam

def test_rwr_is_distribution_and_matches_exact():
    P = H.build_P(ctx.A, sp.csr_matrix((nI, nI)), 0.0).astype(np.float64)
    r, _ = H.power_iteration(P, 3, c=0.15)
    N = P.shape[0]; e = np.zeros(N); e[3] = 1
    exact = 0.15 * sp.linalg.spsolve(sp.identity(N, format="csc") - 0.85 * P.tocsc(), e)
    assert abs(r.sum() - 1) < 1e-8 and np.abs(r - exact).max() < 1e-9

def test_cold_film_reachable_only_through_content():
    films = R.i.unique(); cold = films[:50]
    tr = R[~R.i.isin(cold)]
    c2 = H.Ctx(tr, R[R.i.isin(cold)], nU, nI)
    S_cf = H.rwr_scores(c2, C, 0.0, 0.7); S_h = H.rwr_scores(c2, C, 0.1, 0.7)
    assert S_cf[:, cold].max() == 0                      # collaborative: unreachable
    assert (S_h[:, cold].sum(1) > 0).mean() > 0.9        # hybrid: reachable
    assert H.item_cf_scores(c2)[:, cold].max() == 0

def test_models_beat_popularity():
    pop = H.evaluate(H.popularity_scores(ctx), ctx)["p"]
    icf = H.evaluate(H.item_cf_scores(ctx), ctx)["p"]
    hyb = H.evaluate(H.rwr_scores(ctx, C, 0.1, 0.7), ctx)["p"]
    assert icf > pop + 0.03 and hyb > pop + 0.03, (pop, icf, hyb)

def test_evaluate_perfect_scores():
    S = ctx.rel.astype(np.float32) * 10                    # oracle: hidden liked films on top
    m = H.evaluate(S, ctx)
    assert m["ndcg"] > 0.99

if __name__ == "__main__":
    import scipy.sparse.linalg                              # noqa
    sp.linalg = scipy.sparse.linalg
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try: fn(); print("PASS", name)
            except Exception as ex: fails += 1; print("FAIL", name, repr(ex)[:200])
    sys.exit(1 if fails else 0)
