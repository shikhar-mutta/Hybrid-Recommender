import React from "react";
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, Tooltip, Legend, CartesianGrid, ResponsiveContainer, Cell, ErrorBar } from "recharts";
import { COLORS, pm } from "../shared.js";

export default function Results({ d }) {
  const r = d.results, names = Object.keys(r.main);
  const bestP = Math.max(...names.map((n) => r.main[n].p[0]));
  const bars = names.map((n) => ({ n, p: r.main[n].p[0], pe: r.main[n].p[1], r: r.main[n].r[0], re: r.main[n].r[1] }));
  const lams = Object.keys(r.lam_sweep);
  const lamData = lams.map((l) => ({ lam: l, p: r.lam_sweep[l].p[0], pe: r.lam_sweep[l].p[1], r: r.lam_sweep[l].r[0], tail: r.lam_sweep[l].tail[0], coldR: r.cold_sweep[l].r[0] }));
  const bestLam = lamData.reduce((a, b) => (b.p > a.p ? b : a)).lam;
  const kd = Object.entries(r.k_sweep).map(([k, v]) => ({ k: "k=" + k, p: v.p[0] }));
  const cd = Object.entries(r.c_sweep).map(([c, v]) => ({ c: "c=" + c, p: v.p[0] }));
  return (
    <>
      <div className="card">
        <h3>Accuracy — hide 20% of each user’s ratings (Q1)</h3>
        <p className="sub">precision@10 / recall@10 / NDCG@10, mean ± s.d. over 3 random splits (~600 users each). Hyper-parameters tuned on a separate validation split: c = {r.params.c}, hybrid λ = {r.params.lam}.</p>
        <div className="grid g2">
          <table><thead><tr><th>Method</th><th>precision@10</th><th>recall@10</th><th>NDCG@10</th></tr></thead><tbody>
            {names.map((n) => <tr key={n} className={r.main[n].p[0] === bestP ? "best" : ""}><td><span className="dot" style={{ display: "inline-block", background: COLORS[n], marginRight: 6 }} />{n}</td>
              <td>{pm(r.main[n].p)}</td><td>{pm(r.main[n].r)}</td><td>{pm(r.main[n].ndcg)}</td></tr>)}
          </tbody></table>
          <div style={{ height: 300 }}><ResponsiveContainer>
            <BarChart data={bars} margin={{ bottom: 70 }}><CartesianGrid strokeDasharray="3 3" opacity={0.3} />
              <XAxis dataKey="n" angle={-30} textAnchor="end" interval={0} fontSize={10} height={80} /><YAxis fontSize={11} /><Tooltip /><Legend verticalAlign="top" />
              <Bar isAnimationActive={false} dataKey="p" name="precision@10">{bars.map((b) => <Cell key={b.n} fill={COLORS[b.n]} />)}<ErrorBar dataKey="pe" width={3} stroke="#333" /></Bar>
              <Bar isAnimationActive={false} dataKey="r" name="recall@10" fill="#bbb" />
            </BarChart></ResponsiveContainer></div>
        </div>
      </div>

      <div className="card">
        <h3>The weight of content edges λ (Q5)</h3>
        <p className="sub">λ = probability that a walker on a film follows a content edge instead of a rating edge. <b>Warm accuracy peaks at λ = {bestLam}</b> and collapses when content dominates; for cold films a tiny λ already suffices.</p>
        <div className="grid g3">
          {[["Warm films: precision@10 / recall@10", [["p", "precision@10", "#d62728"], ["r", "recall@10", "#1f77b4"]]],
            ["Share of recs outside top-10% popular", [["tail", "tail share", "#9467bd"]]],
            ["Cold films: recall@10", [["coldR", "cold recall@10", "#2ca02c"]]]].map(([t, lines]) => (
            <div key={t}><div className="note"><b>{t}</b></div><div style={{ height: 240 }}><ResponsiveContainer>
              <LineChart data={lamData} margin={{ left: -10 }}><CartesianGrid strokeDasharray="3 3" opacity={0.3} /><XAxis dataKey="lam" fontSize={10} label={{ value: "λ", position: "insideBottomRight", offset: -4 }} /><YAxis fontSize={10} /><Tooltip />
                {lines.map(([k, n, c]) => <Line isAnimationActive={false} key={k} dataKey={k} name={n} stroke={c} dot={{ r: 3 }} />)}{lines.length > 1 && <Legend />}</LineChart></ResponsiveContainer></div></div>
          ))}
        </div>
        <table style={{ marginTop: 10 }}><thead><tr><th>λ</th>{lamData.map((x) => <th key={x.lam}>{x.lam}</th>)}</tr></thead><tbody>
          <tr><td>precision@10</td>{lamData.map((x) => <td key={x.lam} style={{ fontWeight: x.lam === bestLam ? 700 : 400 }}>{x.p.toFixed(3)}</td>)}</tr>
          <tr><td>tail share</td>{lamData.map((x) => <td key={x.lam}>{x.tail.toFixed(2)}</td>)}</tr>
          <tr><td>cold recall@10</td>{lamData.map((x) => <td key={x.lam}>{x.coldR.toFixed(3)}</td>)}</tr>
        </tbody></table>
      </div>

      <div className="grid g2">
        {[["Number of content neighbours k", kd, "k"], ["Restart probability c", cd, "c"]].map(([t, data, key]) => (
          <div className="card" key={t}><h3>{t}</h3><p className="sub">hybrid precision@10 at λ = {r.params.lam}</p>
            <div style={{ height: 200 }}><ResponsiveContainer><BarChart data={data}><CartesianGrid strokeDasharray="3 3" opacity={0.3} /><XAxis dataKey={key} fontSize={11} /><YAxis domain={[0.14, 0.2]} fontSize={11} /><Tooltip /><Bar isAnimationActive={false} dataKey="p" fill="#d62728" /></BarChart></ResponsiveContainer></div></div>
        ))}
      </div>
    </>
  );
}
