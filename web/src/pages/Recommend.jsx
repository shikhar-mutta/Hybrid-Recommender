import React, { useEffect, useMemo, useRef, useState } from "react";
import { forceSimulation, forceLink, forceManyBody, forceCenter, forceCollide, forceX, forceY } from "d3-force";
import { COLORS } from "../shared.js";

const METHODS = ["Item-based CF", "RWR (ratings only)", "Content only", "Hybrid RWR", "Hybrid + popularity discount", "node2vec"];

function FilmRow({ f, i, films, hl, setHl, tail }) {
  return (
    <div className={"film" + (hl === i ? " hl" : "")} onMouseEnter={() => setHl(i)} onMouseLeave={() => setHl(null)}>
      <span className="pop">{films.pop[i]}★</span>
      {films.title[i]} {tail && <span className="tag tail">long tail</span>}
      <small>{films.genres[i].replaceAll("|", " · ")}</small>
    </div>
  );
}

function Graph({ user, films, method }) {
  const ref = useRef();
  const [nodes, setNodes] = useState([]);
  const [links, setLinks] = useState([]);
  const [hover, setHover] = useState(null);
  const W = 900, H = 600;
  const { nodeSpec, linkSpec } = useMemo(() => {
    const fav = user.fav.slice(0, 7), rec = user.recs[method];
    const ns = [{ id: "u", kind: "user", label: `user ${user.id}` }];
    const seen = new Set();
    fav.forEach((i) => { seen.add(i); ns.push({ id: "f" + i, film: i, kind: "liked" }); });
    rec.forEach((i) => { if (!seen.has(i)) { seen.add(i); ns.push({ id: "f" + i, film: i, kind: "rec" }); } else ns.find((n) => n.film === i).kind = "both"; });
    const ls = [];
    fav.forEach((i) => ls.push({ source: "u", target: "f" + i, kind: "rating" }));
    const ids = new Set(ns.map((n) => n.id));
    ns.filter((n) => n.film != null).forEach((n) => {
      const nb = films.nb[n.film];
      nb.c.forEach((j) => { if (ids.has("f" + j) && n.film < j + 1e9) ls.push({ source: n.id, target: "f" + j, kind: "content" }); });
      nb.r.forEach((j) => { if (ids.has("f" + j)) ls.push({ source: n.id, target: "f" + j, kind: "cf" }); });
    });
    const rec2 = ns.filter((n) => n.kind === "rec");
    rec2.forEach((n) => { if (!ls.some((l) => l.target === n.id || l.source === n.id)) ls.push({ source: "u", target: n.id, kind: "weak" }); });
    return { nodeSpec: ns, linkSpec: ls };
  }, [user, films, method]);

  useEffect(() => {
    const ns = nodeSpec.map((n) => ({ ...n }));
    const ls = linkSpec.map((l) => ({ ...l }));
    const sim = forceSimulation(ns)
      .force("link", forceLink(ls).id((d) => d.id).distance((l) => (l.kind === "rating" ? 170 : l.kind === "weak" ? 260 : 120)).strength(0.5))
      .force("charge", forceManyBody().strength(-650))
      .force("center", forceCenter(W / 2, H / 2))
      .force("x", forceX(W / 2).strength(0.07)).force("y", forceY(H / 2).strength(0.09))
      .force("collide", forceCollide(34));
    sim.on("tick", () => { ns.forEach((n) => { n.x = Math.max(60, Math.min(W - 60, n.x)); n.y = Math.max(30, Math.min(H - 30, n.y)); }); setNodes([...ns]); setLinks([...ls]); });
    return () => sim.stop();
  }, [nodeSpec, linkSpec]);

  const col = { user: "#444", liked: "#1f77b4", rec: COLORS[method] || "#d62728", both: "#2ca02c" };
  const lc = { rating: "#1f77b4", content: "#d62728", cf: "#ff7f0e", weak: "#bbb" };
  const short = (t) => (t.length > 26 ? t.slice(0, 24) + "…" : t);
  return (
    <div>
      <svg ref={ref} viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxHeight: 640 }}>
        {links.map((l, i) => (
          <line key={i} x1={l.source.x} y1={l.source.y} x2={l.target.x} y2={l.target.y} stroke={lc[l.kind]}
            strokeWidth={l.kind === "rating" ? 2 : 1.4} strokeDasharray={l.kind === "content" ? "5 3" : l.kind === "weak" ? "1 4" : undefined} opacity={0.75} />
        ))}
        {nodes.map((n) => (
          <g key={n.id} transform={`translate(${n.x},${n.y})`} onMouseEnter={() => setHover(n)} onMouseLeave={() => setHover(null)}>
            <circle r={n.kind === "user" ? 13 : 8} fill={col[n.kind]} stroke="#fff" strokeWidth="1.5" />
            <text y={n.kind === "user" ? 26 : -12} textAnchor="middle" fontSize="11" stroke="var(--card)" strokeWidth="3" paintOrder="stroke">{n.kind === "user" ? n.label : short(films.title[n.film])}</text>
          </g>
        ))}
      </svg>
      <div className="note">
        <span className="chip" style={{ borderLeft: "4px solid #444" }}>user</span>
        <span className="chip" style={{ borderLeft: "4px solid #1f77b4" }}>film the user liked</span>
        <span className="chip" style={{ borderLeft: `4px solid ${col.rec}` }}>recommended ({method})</span>
        <span className="chip" style={{ borderLeft: "4px solid #2ca02c" }}>liked & recommended</span><br />
        Edges: <b style={{ color: "#1f77b4" }}>─ rating</b> · <b style={{ color: "#d62728" }}>- - content similarity</b> · <b style={{ color: "#ff7f0e" }}>─ item–item CF similarity</b> · dotted: no link to the user’s liked films (reached by the walk through other users).
        {hover && hover.film != null && <div style={{ marginTop: 4 }}><b>{films.title[hover.film]}</b> — {films.genres[hover.film]} — {films.pop[hover.film]} ratings</div>}
      </div>
    </div>
  );
}

export default function Recommend({ d, api }) {
  const { films, users } = d;
  const [uid, setUid] = useState(105);
  const [hl, setHl] = useState(null);
  const [gm, setGm] = useState("Hybrid RWR");
  const user = users[uid - 1];
  const top10 = useMemo(() => {
    const ord = films.pop.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, Math.ceil(0.1 * films.pop.filter((p) => p > 0).length));
    return new Set(ord.map((x) => x[1]));
  }, [films]);
  const [live, setLive] = useState({ lam: 0.1, c: 0.7, beta: 0 });
  const [liveRecs, setLiveRecs] = useState(null);
  const [busy, setBusy] = useState(false);
  const run = () => {
    setBusy(true);
    fetch(`/api/recommend?user=${uid}&lam=${live.lam}&c=${live.c}&beta=${live.beta}`).then((r) => r.json())
      .then((j) => setLiveRecs(j.recs || null)).finally(() => setBusy(false));
  };
  useEffect(() => { setLiveRecs(null); }, [uid]);
  const rnd = () => setUid(1 + Math.floor(Math.random() * users.length));

  return (
    <>
      <div className="card">
        <div className="row">
          <label className="s">User id (1–{users.length})
            <input type="number" min="1" max={users.length} value={uid} onChange={(e) => setUid(Math.min(users.length, Math.max(1, +e.target.value || 1)))} />
          </label>
          <button className="btn sec" onClick={rnd}>Random user</button>
          <button className="btn sec" onClick={() => setUid(105)} title="liked both 3 Idiots and Taare Zameen Par">User 105 (brief’s example)</button>
          <span className="note">{user.n_rated} ratings · {user.n_liked} liked (≥ 4★) · taste community C{user.community}. Recommendations are computed with all ratings; films the user already rated are excluded.</span>
        </div>
        <div style={{ marginTop: 10 }}><b>Favourites:</b>{" "}{user.fav.map((i) => <span key={i} className="chip">{films.title[i]}</span>)}</div>
      </div>

      <div className="card">
        <h3>Recommendations side by side</h3>
        <p className="sub">Hover a film to see where else it appears. ★ = number of ratings in the data (popularity). “long tail” = outside the 10% most-rated films.</p>
        <div className="cols">
          {METHODS.map((m) => (
            <div className="col" key={m}>
              <h4><span className="dot" style={{ background: COLORS[m] }} />{m}</h4>
              {user.recs[m].map((i) => <FilmRow key={i} f={i} i={i} films={films} hl={hl} setHl={setHl} tail={!top10.has(i)} />)}
            </div>
          ))}
          {liveRecs && (
            <div className="col"><h4><span className="dot" style={{ background: "#000" }} />Live (λ={live.lam}, c={live.c}, β={live.beta})</h4>
              {liveRecs.map((f) => <div key={f.id} className={"film" + (hl === f.id ? " hl" : "")} onMouseEnter={() => setHl(f.id)} onMouseLeave={() => setHl(null)}><span className="pop">{f.pop}★</span>{f.title}<small>{f.genres.replaceAll("|", " · ")}</small></div>)}
            </div>
          )}
        </div>
      </div>

      <div className="card">
        <h3>Try the hybrid live</h3>
        <p className="sub">Re-runs the random walk for this user with your own settings (needs <code>python3 server/api.py</code>). λ = weight of content edges against rating edges; c = restart probability; β = popularity discount. Note: a large β together with content edges favours obscure films (it lowers accuracy), see the report.</p>
        {!api && <div className="warn">Live API is offline. Start it with <code>python3 server/api.py</code> in the project folder.</div>}
        <div className="row" style={{ opacity: api ? 1 : 0.5 }}>
          {[["lam", "λ content weight", 0, 1, 0.01], ["c", "c restart prob.", 0.05, 0.95, 0.05], ["beta", "β popularity discount", 0, 1, 0.05]].map(([k, lab, lo, hi, st]) => (
            <label className="s" key={k}>{lab}: <b>{live[k]}</b>
              <input type="range" min={lo} max={hi} step={st} value={live[k]} onChange={(e) => setLive({ ...live, [k]: +e.target.value })} disabled={!api} />
            </label>
          ))}
          <button className="btn" onClick={run} disabled={!api || busy}>{busy ? "Running…" : "Run walk"}</button>
        </div>
      </div>

      <div className="card">
        <div className="row"><h3 style={{ margin: 0 }}>Graph explorer</h3>
          <select value={gm} onChange={(e) => setGm(e.target.value)}>{METHODS.map((m) => <option key={m}>{m}</option>)}</select>
        </div>
        <p className="sub">The user’s liked films and the films recommended by the selected method; links show why they belong together (content similarity vs. similar raters).</p>
        <Graph user={user} films={films} method={gm} />
      </div>
    </>
  );
}
