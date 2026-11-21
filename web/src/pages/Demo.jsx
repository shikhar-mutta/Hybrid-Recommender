import MultiSelect from "../MultiSelect.jsx";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, CartesianGrid, ResponsiveContainer, ReferenceLine } from "recharts";
import { COLORS } from "../shared.js";
import { METHODS, FilmRow } from "./Recommend.jsx";

const PRESETS = {
  "Toy Story 4 (2019)": { title: "Toy Story 4", genres: ["Adventure", "Animation", "Children", "Comedy", "Fantasy"], tags: ["pixar", "toys", "fun"] },
  "John Wick: Chapter 3 (2019)": { title: "John Wick Chapter 3 Parabellum", genres: ["Action", "Crime", "Thriller"], tags: ["assassin", "hitman", "gun fu", "keanu reeves"] },
  "Dangal (2016)": { title: "Dangal", genres: ["Action", "Drama"], tags: ["wrestling", "sports", "father daughter", "india", "inspirational", "bollywood"] },
};
const Step = ({ n, title, children }) => (
  <div className="card">
    <h3><span className="chip" style={{ background: "var(--acc)", color: "#fff", fontWeight: 700 }}>Step {n}</span> {title}</h3>
    {children}
  </div>
);

export default function Demo({ d, api }) {
  const { films, users } = d;
  const [uid, setUid] = useState(105);
  const [hl, setHl] = useState(null);
  const user = users[uid - 1];
  const top10 = useMemo(() => {
    const k = Math.ceil(0.1 * films.pop.filter((p) => p > 0).length);
    return new Set(films.pop.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, k).map((x) => x[1]));
  }, [films]);

  const [f, setF] = useState({ ...PRESETS["Toy Story 4 (2019)"], lam: 0.1 });
  const [out, setOut] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const addFilm = () => {
    setBusy(true); setErr(null);
    fetch("/api/newfilm", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...f, genres: f.genres.join(", "), tags: f.tags.join(", ") }) })
      .then((r) => r.json()).then((j) => (j.error ? setErr(j.error) : setOut(j))).catch((e) => setErr(String(e))).finally(() => setBusy(false));
  };
  useEffect(() => { if (api && location.hash.includes("run=1") && !out && !busy) addFilm(); }, [api]);   // deep link: #tab=0&run=1
  const myPct = out ? out.pct[uid - 1] : null;
  const myRank = out ? 1 + out.pct.filter((p) => p < myPct).length : null;


  // ---- Step 4: the weight of content --------------------------------------------------
  const R = d.results;
  const lams = Object.keys(R.lam_sweep);                       // "0.0","0.02",...,"1.0"
  const curve = lams.map((l) => ({ lam: l, p: R.lam_sweep[l].p[0], r: R.lam_sweep[l].r[0], tail: R.lam_sweep[l].tail[0], coldR: R.cold_sweep[l].r[0] }));
  const peak = curve.reduce((a, b) => (b.p > a.p ? b : a));
  const [li, setLi] = useState(lams.indexOf(peak.lam));
  const cur = curve[li];
  const [byLam, setByLam] = useState({});                      // lam -> recs for the picked user
  const cache = useRef({});
  useEffect(() => {
    if (!api) return;
    const need = [...new Set(["0.0", cur.lam, "1.0"])];
    const t = setTimeout(() => {
      need.forEach((l) => {
        const key = `${uid}|${l}`;
        if (cache.current[key]) { setByLam((m) => ({ ...m, [l]: cache.current[key] })); return; }
        fetch(`/api/recommend?user=${uid}&lam=${l}&c=${R.params.c}&beta=0`).then((r) => r.json())
          .then((j) => { if (j.recs) { cache.current[key] = j.recs; setByLam((m) => ({ ...m, [l]: j.recs })); } });
      });
    }, 350);
    return () => clearTimeout(t);
  }, [api, uid, cur.lam]);
  const base = new Set((byLam["0.0"] || []).map((f) => f.id));
  const overlap = (l) => (byLam[l] || []).filter((f) => base.has(f.id)).length;
  const tailShare = (l) => (byLam[l] || []).length ? (byLam[l].filter((f) => !top10.has(f.id)).length / byLam[l].length) : null;
  const cols4 = [["0.0", "λ = 0 — ratings only"], [cur.lam, `λ = ${cur.lam} — your slider`], ["1.0", "λ = 1 — content only"]];

  return (
    <>
      <div className="card" style={{ background: "color-mix(in srgb,var(--acc) 6%,var(--card))" }}>
        <b>Demo.</b> 1 · pick a user → 2 · see the recommendations of each method side by side → 3 · add a new film with <i>no ratings</i> and see which users the hybrid recommends it to → 4 · the weight of content λ: at what λ does accuracy peak?
      </div>

      <Step n={1} title="Pick a user">
        <div className="row">
          <label className="s">User id (1–{users.length})
            <input type="number" min="1" max={users.length} value={uid} onChange={(e) => setUid(Math.min(users.length, Math.max(1, +e.target.value || 1)))} />
          </label>
          <button className="btn sec" onClick={() => setUid(1 + Math.floor(Math.random() * users.length))}>Random user</button>
          <button className="btn sec" onClick={() => setUid(105)} title="liked both 3 Idiots and Taare Zameen Par">User 105 (3 Idiots & Taare Zameen Par fan)</button>
          <span className="note">{user.n_rated} ratings · {user.n_liked} liked (≥ 4★) · taste community C{user.community}</span>
        </div>
        <div style={{ marginTop: 10 }}><b>Favourites:</b> {user.fav.map((i) => <span key={i} className="chip">{films.title[i]}</span>)}</div>
      </Step>

      <Step n={2} title={`Recommendations of each method for user ${uid}, side by side`}>
        <p className="sub">Films the user already rated are excluded. Hover a film to see where else it appears. ★ = number of ratings (popularity); “long tail” = outside the 10% most-rated films.</p>
        <div className="cols">
          {METHODS.map((m) => (
            <div className="col" key={m}>
              <h4><span className="dot" style={{ background: COLORS[m] }} />{m}</h4>
              {user.recs[m].map((i) => <FilmRow key={i} f={i} i={i} films={films} hl={hl} setHl={setHl} tail={!top10.has(i)} />)}
            </div>
          ))}
        </div>
      </Step>

      <Step n={3} title="Add a new film with no ratings — which users does the hybrid recommend it to?">
        <p className="sub">A new film has no rating edge, so item-based CF and the ratings-only walk give it a score of exactly <b>0 for every user</b> and can never recommend it.
          The hybrid embeds its text, links it to its 20 most similar films and runs the walk from every user. Users are ranked by where the film lands on their shelf of <i>new releases</i> (films from 2015 on they have not seen): 1% = first.</p>
        {!api && <div className="warn" style={{ marginBottom: 10 }}>Live API is offline — start it with <code>./run.sh api</code> (or <code>python3 server/api.py</code>) and refresh. Steps 1 and 2 work without it.</div>}
        <div className="row" style={{ marginBottom: 10 }}>
          <span className="note">Examples:</span>
          {Object.keys(PRESETS).map((k) => <button key={k} className="btn sec" onClick={() => { setF({ ...PRESETS[k], lam: f.lam }); setOut(null); }}>{k}</button>)}
        </div>
        <div className="grid g3">
          <label className="s">Title<input value={f.title} onChange={(e) => setF({ ...f, title: e.target.value })} /></label>
          <MultiSelect label="Genres" options={d.vocab.genres} value={f.genres} onChange={(v) => setF({ ...f, genres: v })} placeholder="Choose genres…" />
          <MultiSelect label="Tags" options={d.vocab.tags} value={f.tags} onChange={(v) => setF({ ...f, tags: v })} placeholder="Choose or type tags…" max={12} allowCustom />
        </div>
        <div className="row" style={{ marginTop: 10 }}>
          <label className="s">λ content weight: <b>{f.lam}</b><input type="range" min="0.01" max="0.5" step="0.01" value={f.lam} onChange={(e) => setF({ ...f, lam: +e.target.value })} /></label>
          <button className="btn" onClick={addFilm} disabled={!api || busy || f.genres.length === 0}>{busy ? "Running the walk…" : "Add film & recommend to users"}</button>
        </div>
        {err && <div className="warn" style={{ marginTop: 10 }}>{err}</div>}

        {out && (
          <div style={{ marginTop: 16 }}>
            <div className="grid g3">
              <div className="card kpi" style={{ marginBottom: 0 }}><small>Item-CF / ratings-only walk</small><b>0 of {users.length} users</b><small>cannot reach a film without ratings</small></div>
              <div className="card kpi" style={{ marginBottom: 0 }}><small>Hybrid (λ = {f.lam})</small><b>{out.reached} of {users.length} users</b><small>reached through content edges</small></div>
              <div className="card kpi" style={{ marginBottom: 0 }}><small>For the user you picked (user {uid})</small><b>{myPct}% · rank {myRank} of {users.length}</b><small>percentile on their new-release shelf · rank among users</small></div>
            </div>
            <div className="grid g2" style={{ marginTop: 16 }}>
              <div>
                <h4 style={{ margin: "0 0 6px" }}>Why: content neighbours the new film was linked to</h4>
                <table><thead><tr><th>Film</th><th>Genres</th><th>cosine</th></tr></thead><tbody>
                  {out.neighbours.map((n) => <tr key={n.title}><td>{n.title}</td><td style={{ textAlign: "left" }}>{n.genres.replaceAll("|", " · ")}</td><td>{n.cos}</td></tr>)}
                </tbody></table>
              </div>
              <div>
                <h4 style={{ margin: "0 0 6px" }}>Users the hybrid recommends it to (top 15)</h4>
                <table><thead><tr><th>User</th><th>shelf %</th><th>their favourites</th><th></th></tr></thead><tbody>
                  {out.users.map((u) => (
                    <tr key={u.user} className={u.user === uid ? "best" : ""}>
                      <td>{u.user}</td><td>{u.percentile}%</td><td style={{ textAlign: "left" }}>{u.fav.join("; ")}</td>
                      <td><button className="btn sec" style={{ padding: "2px 8px" }} onClick={() => { setUid(u.user); window.scrollTo({ top: 0, behavior: "smooth" }); }}>see user</button></td>
                    </tr>))}
                </tbody></table>
                <p className="note">“see user” loads that user into Step 1 so you can compare with their recommendations in Step 2.</p>
              </div>
            </div>
          </div>
        )}
      </Step>

      <Step n={4} title="The weight of content — at what λ does accuracy peak?">
        <p className="sub">λ is the probability that the walker, standing on a film, follows a <b>content edge</b> instead of a <b>rating edge</b> (λ = 0: ratings only, λ = 1: content only).
          Measured on hidden ratings (precision@10, mean of 3 random splits). <b>Accuracy peaks at λ = {peak.lam}</b> ({peak.p.toFixed(3)} vs {curve[0].p.toFixed(3)} at λ = 0, better in all 3 splits) and collapses when content dominates; for films without ratings a tiny λ already suffices.</p>
        <div className="row" style={{ marginBottom: 8 }}>
          <label className="s" style={{ minWidth: 320 }}>Move the slider — λ = <b>{cur.lam}</b>
            <input type="range" min="0" max={lams.length - 1} step="1" value={li} onChange={(e) => setLi(+e.target.value)} />
          </label>
          <button className="btn sec" onClick={() => setLi(lams.indexOf(peak.lam))}>Jump to the peak (λ = {peak.lam})</button>
          <button className="btn sec" onClick={() => setLi(0)}>λ = 0</button>
        </div>
        <div className="grid g3" style={{ marginBottom: 12 }}>
          <div className="card kpi" style={{ marginBottom: 0 }}><small>precision@10 · films with ratings</small><b style={{ color: cur.lam === peak.lam ? "#2ca02c" : undefined }}>{cur.p.toFixed(3)}</b><small>{cur.lam === peak.lam ? "the peak" : `${(cur.p - peak.p >= 0 ? "+" : "")}${(cur.p - peak.p).toFixed(3)} vs peak · ${(cur.p - curve[0].p >= 0 ? "+" : "")}${(cur.p - curve[0].p).toFixed(3)} vs λ = 0`}</small></div>
          <div className="card kpi" style={{ marginBottom: 0 }}><small>share of recommendations outside the top-10% popular films</small><b>{(cur.tail * 100).toFixed(0)}%</b><small>more content → more obscure films</small></div>
          <div className="card kpi" style={{ marginBottom: 0 }}><small>recall@10 · films with no ratings (cold start)</small><b>{cur.coldR.toFixed(3)}</b><small>{cur.lam === "0.0" ? "= chance: CF cannot reach them" : "reached through content edges"}</small></div>
        </div>
        <div className="grid g3">
          {[["Films with ratings: precision@10 / recall@10", [["p", "precision@10", "#d62728"], ["r", "recall@10", "#1f77b4"]]],
            ["Share of recs outside top-10% popular", [["tail", "tail share", "#9467bd"]]],
            ["Films without ratings: recall@10", [["coldR", "cold recall@10", "#2ca02c"]]]].map(([t, lines]) => (
            <div key={t}><div className="note"><b>{t}</b></div><div style={{ height: 230 }}><ResponsiveContainer>
              <LineChart data={curve} margin={{ left: -10 }}><CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis dataKey="lam" fontSize={10} /><YAxis fontSize={10} /><Tooltip />
                <ReferenceLine x={peak.lam} stroke="#2ca02c" strokeDasharray="4 3" label={{ value: "peak", fontSize: 10, fill: "#2ca02c", position: "top" }} />
                <ReferenceLine x={cur.lam} stroke="#000" strokeWidth={2} />
                {lines.map(([k, n, c]) => <Line key={k} dataKey={k} name={n} stroke={c} isAnimationActive={false} dot={{ r: 3 }} />)}{lines.length > 1 && <Legend />}
              </LineChart></ResponsiveContainer></div></div>
          ))}
        </div>
        <h4 style={{ margin: "16px 0 6px" }}>What the slider does to user {uid}’s recommendations</h4>
        {!api && <div className="warn">Start the live API (<code>./run.sh api</code>) to see the recommendation lists at different λ. The charts above work without it.</div>}
        {api && (
          <div className="cols" style={{ gridTemplateColumns: "repeat(auto-fit,minmax(280px,1fr))" }}>
            {cols4.map(([l, name], k) => (
              <div className="col" key={k}>
                <h4>{name}</h4>
                <div className="note" style={{ marginBottom: 4 }}>
                  {byLam[l] ? `${overlap(l)}/10 same as λ = 0 · ${(tailShare(l) * 100).toFixed(0)}% long tail` : "loading…"}
                </div>
                {(byLam[l] || []).map((f) => <FilmRow key={f.id} f={f.id} i={f.id} films={films} hl={hl} setHl={setHl} tail={!top10.has(f.id)} />)}
              </div>
            ))}
          </div>
        )}
        <p className="note">Reading it: at λ ≈ 0.1 the list is almost the ratings-only list with a few content-driven changes (best accuracy). Pushed to λ = 1 the list is all obscure films similar in text to the user’s favourites, and accuracy is near zero.</p>
      </Step>
    </>
  );
}
