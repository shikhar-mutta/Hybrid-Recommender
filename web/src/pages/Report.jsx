import React from "react";

const FIGS = [["fig_data", "Data: ratings and degree distributions"], ["fig_warmup", "Warm-up: personalised PageRank"], ["fig_accuracy", "Accuracy"], ["fig_lambda", "Weight of content λ"],
  ["fig_cold", "Cold start"], ["fig_popularity", "Popularity"], ["fig_tradeoff", "Accuracy vs novelty"], ["fig_communities", "Communities of taste"]];
export default function Report() {
  return (
    <>
      <div className="card"><div className="row"><h3 style={{ margin: 0 }}>LaTeX report</h3><a className="btn" href="/report/report.pdf" target="_blank" style={{ textDecoration: "none", padding: "7px 12px", borderRadius: 7, color: "#fff" }}>Open PDF</a></div>
        <iframe title="report" src="/report/report.pdf" style={{ width: "100%", height: 720, border: "1px solid var(--line)", borderRadius: 8, marginTop: 12, background: "#fff" }} /></div>
      <div className="card"><h3>Figures from the notebook</h3><div className="imgs">
        {FIGS.map(([f, t]) => <figure key={f} style={{ margin: 0 }}><img src={`/report/${f}.png`} alt={t} /><figcaption className="note">{t}</figcaption></figure>)}</div></div>
    </>
  );
}
