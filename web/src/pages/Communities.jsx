import React from "react";

const color = (v) => {   // log2 lift -> diverging colour
  const x = Math.max(-1, Math.min(1, Math.log2(Math.max(v, 1e-3))));
  return x >= 0 ? `rgba(214,39,40,${x * 0.9})` : `rgba(31,119,180,${-x * 0.9})`;
};
export default function Communities({ d }) {
  const c = d.communities;
  return (
    <>
      <div className="card">
        <h3>Communities of taste (Q4)</h3>
        <p className="sub">The user–film graph projected onto users (cosine of mean-centred ratings, 15 nearest users each) and split with Louvain. Communities are found from ratings only — never from genres.</p>
        <div className="grid g3">
          <div className="card kpi"><small>communities / modularity</small><b>{c.sizes.length} / {c.modularity}</b><small>sizes {c.sizes.join(", ")}</small></div>
          <div className="card kpi"><small>between-community genre spread</small><b>{c.spread}</b><small>random relabelling: {c.null_mean} ± {c.null_sd}</small></div>
          <div className="card kpi"><small>permutation p-value (2000 shuffles)</small><b>{c.p}</b><small>communities differ strongly in genre mix</small></div>
        </div>
        <table className="hm" style={{ marginTop: 8 }}>
          <thead><tr><th></th>{c.genres.map((g) => <th key={g} style={{ writingMode: "vertical-rl", height: 90, textAlign: "left" }}>{g}</th>)}</tr></thead>
          <tbody>{c.communities.map((k) => (
            <tr key={k.id}><td style={{ textAlign: "left", paddingRight: 8, whiteSpace: "nowrap", fontSize: 12 }}>C{k.id} (n={k.size})</td>
              {k.lift.map((v, i) => <td key={i} title={`${c.genres[i]}: ${v.toFixed(2)}×`} style={{ background: color(v), height: 30, minWidth: 28 }}>{v >= 1.5 || v <= 0.6 ? v.toFixed(1) : ""}</td>)}</tr>))}
          </tbody>
        </table>
        <p className="note">Cell colour = genre share of the community’s liked films relative to all users (red = over-represented, blue = under-represented; numbers show ratio when strong).
          Communities are not simply “one genre each”: almost every user’s top genre is Drama or Comedy — the differences are in the secondary genres.</p>
      </div>
    </>
  );
}
