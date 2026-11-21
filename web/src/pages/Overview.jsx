import React from "react";
import { fmt } from "../shared.js";

export default function Overview({ d, go }) {
  const r = d.results, m = r.main, ds = r.dataset;
  const best = (r.lam_sweep["0.1"].p[0] - r.lam_sweep["0.0"].p[0]);
  const cold = r.cold_main;
  return (
    <>
      <div className="card">
        <h3>What this is</h3>
        <p className="sub" style={{ marginBottom: 0 }}>
          Users, films and content in <b>one graph</b>: ratings link users to films, sentence-embedding similarity links films to films.
          A <b>random walk with restart</b> (personalised PageRank) reaches films that similar users liked <i>and</i> films similar to those.
          This interface shows the recommendations of every method side by side, the effect of the weight of content edges λ,
          the cold-start behaviour for films without ratings, and the results of all investigations (mean over 3 random splits).
        </p>
      </div>
      <div className="grid g3">
        <div className="card kpi"><small>Data</small><b>{ds.users} users · {ds.films.toLocaleString()} films</b><small>{ds.ratings.toLocaleString()} ratings · density {(ds.density * 100).toFixed(1)}%</small></div>
        <div className="card kpi"><small>precision@10 — item-CF → hybrid (λ=0.1)</small><b>{fmt(m["Item-based CF"].p[0])} → {fmt(m["Hybrid RWR"].p[0])}</b><small>content weight gain {best >= 0 ? "+" : ""}{fmt(best, 4)} vs. ratings-only walk</small></div>
        <div className="card kpi"><small>cold-start recall@10 — chance → hybrid</small><b>{fmt(cold["Random"].r[0])} → {fmt(cold["Hybrid RWR"].r[0])}</b><small>{(cold["Hybrid RWR"].r[0] / cold["Random"].r[0]).toFixed(1)}× chance; CF methods are at chance</small></div>
        <div className="card kpi"><small>best warm accuracy</small><b>{fmt(m["RWR + popularity discount"].p[0])}</b><small>popularity-discounted walk (β = {r.params.beta_cf})</small></div>
        <div className="card kpi"><small>recommendations outside top-10% popular films</small><b>{(m["Hybrid RWR"].tail[0] * 100).toFixed(1)}%</b><small>hybrid; item-CF {(m["Item-based CF"].tail[0] * 100).toFixed(1)}%, RWR {(m["RWR (ratings only)"].tail[0] * 100).toFixed(1)}%</small></div>
        <div className="card kpi"><small>communities of taste</small><b>{d.communities.communities.length} (Q = {d.communities.modularity})</b><small>genre structure permutation p = {d.communities.p}</small></div>
      </div>
      <div className="card">
        <h3>Where to look</h3>
        <div className="row">
          {[[0, "Demo: pick a user → all methods side by side → add a new film"], [2, "Recommendations: graph explorer and live walk"], [3, "New film (cold start)"], [4, "Accuracy and the weight of content λ"], [5, "Cold start and popularity"], [6, "Communities of taste"]].map(([i, t]) =>
            <button key={i} className="btn sec" onClick={() => go(i)}>{t}</button>)}
        </div>
      </div>
    </>
  );
}
