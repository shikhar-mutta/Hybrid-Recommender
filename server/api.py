"""Live API for the web interface (stdlib only).   python3 server/api.py   ->  http://localhost:8000

GET  /api/health
GET  /api/recommend?user=105&lam=0.1&c=0.7&beta=0.0     hybrid recommendations for one user, live
POST /api/newfilm  {"title","genres","tags","lam"}        which users should get a film with no ratings
"""
import os, sys, json, pickle, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import numpy as np, scipy.sparse as sp
import hybrid_rec as H

R, M, T = H.load_data()
nU, nI = int(R.u.max()) + 1, len(M)
E = np.load("results/emb.npy")
P0 = pickle.load(open("results/seed0.pkl", "rb"))["params"]
C = H.knn_graph(E, P0["k"])
ctx = H.Ctx(R, R.iloc[:0], nU, nI)
title, genres = M.title.values, M.genres.values
yr = M.title.str.extract(r"\((\d{4})\)\s*$")[0].astype(float).values
recent = np.where(yr >= 2015)[0]
_st, lock = None, threading.Lock()
print("API ready: %d users, %d films" % (nU, nI), flush=True)


def encoder():
    global _st
    if _st is None:
        from sentence_transformers import SentenceTransformer
        _st = SentenceTransformer("all-MiniLM-L6-v2")
    return _st


def film_json(i):
    return {"id": int(i), "title": title[i], "genres": genres[i], "pop": int(ctx.pop[i])}


def recommend(user, lam, c, beta):
    u = user - 1
    P = H.build_P(ctx.A, C, lam)
    S = H.rwr_all(P, nU, c, users=[u])
    S = H.debias(S, ctx.pop, beta) if beta else S
    full = np.zeros((nU, nI), np.float32); full[u] = S[0]
    top = H.top_k(full, ctx, 10)[u]
    return [film_json(i) for i in top]


def newfilm(ttl, gen, tags, lam, top_users=15):
    text = f"{ttl}. Genres: {gen}. Tags: {tags}"
    with lock:
        e = encoder().encode([text], normalize_embeddings=True).astype(np.float32)
    E2 = np.vstack([E, e]); C2 = H.knn_graph(E2, P0["k"])
    A2 = sp.hstack([ctx.A, sp.csr_matrix((nU, 1), dtype=np.float32)], format="csr")
    S = H.rwr_all(H.build_P(A2, C2, lam), nU, P0["c"])
    S[ctx.seen_r, ctx.seen_c] = -np.inf
    pool = np.append(recent, nI); Sp = S[:, pool]; s_new = Sp[:, -1]
    pct = ((Sp > s_new[:, None]).sum(1) + 1) / np.isfinite(Sp).sum(1) * 100
    order = np.argsort(pct)[:top_users]
    nb = np.argsort(-C2[nI].toarray().ravel())[:6]
    out = []
    for u in order:
        L = ctx.Liked[u]; fav = L.indices[np.argsort(-L.data)][:4]
        out.append({"user": int(u + 1), "percentile": round(float(pct[u]), 2), "fav": [title[i] for i in fav]})
    return {"neighbours": [{"title": title[i], "genres": genres[i], "cos": round(float(C2[nI, i]), 3)} for i in nb],
            "reached": int((s_new > 0).sum()), "users": out, "shelf": int(len(recent))}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        for k, v in [("Content-Type", "application/json"), ("Access-Control-Allow-Origin", "*"),
                     ("Access-Control-Allow-Headers", "Content-Type"), ("Content-Length", str(len(b)))]:
            self.send_header(k, v)
        self.end_headers(); self.wfile.write(b)

    def do_OPTIONS(self): self._send(200, {})

    def do_GET(self):
        url = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/api/health":
                return self._send(200, {"ok": True, "users": nU, "films": nI, "params": P0})
            if url.path == "/api/recommend":
                user = int(q.get("user", 1))
                assert 1 <= user <= nU
                lam = min(max(float(q.get("lam", P0["lam"])), 0), 1)
                c = min(max(float(q.get("c", P0["c"])), 0.05), 0.95)
                beta = min(max(float(q.get("beta", 0)), 0), 1)
                return self._send(200, {"user": user, "lam": lam, "c": c, "beta": beta, "recs": recommend(user, lam, c, beta)})
        except Exception as ex:
            return self._send(400, {"error": str(ex)})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if urlparse(self.path).path == "/api/newfilm":
                ttl = str(body.get("title", "New film"))[:120]
                return self._send(200, newfilm(ttl, str(body.get("genres", ""))[:200], str(body.get("tags", ""))[:300],
                                               min(max(float(body.get("lam", P0["lam"])), 0.001), 1)))
        except Exception as ex:
            return self._send(400, {"error": str(ex)})
        self._send(404, {"error": "not found"})

    def log_message(self, *a): pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8000), Handler).serve_forever()
