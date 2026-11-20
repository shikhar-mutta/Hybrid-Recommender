import React from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Cell, ErrorBar } from "recharts";
import { COLORS, pm } from "../shared.js";

export default function Cold({ d }) {
  const r = d.results, cn = Object.keys(r.cold_main), pn = Object.keys(r.main);
  const cb = cn.map((n) => ({ n, r: r.cold_main[n].r[0], e: r.cold_main[n].r[1], lift: r.cold_main[n].user_lift[0], le: r.cold_main[n].user_lift[1] }));
  const pb = pn.map((n) => ({ n, tail: r.main[n].tail[0], cov: r.main[n].coverage[0] }));
  return (
    <>
      <div className="card">
        <h3>Cold start — all ratings of {r.n_cold} films ({"10%"}) hidden (Q2)</h3>
        <p className="sub">Left: rank the cold films for each user (recall@10; random is the chance level). Right: for each cold film rank users; lift of precision@10 over the film’s base rate of likers (random ≈ 1).</p>
        <div className="grid g2">
          {[["r", "e", "recall@10 (cold films per user)"], ["lift", "le", "user-side lift"]].map(([k, e, t]) => (
            <div key={k}><div className="note"><b>{t}</b></div><div style={{ height: 270 }}><ResponsiveContainer>
              <BarChart data={cb} margin={{ bottom: 70 }}><CartesianGrid strokeDasharray="3 3" opacity={0.3} /><XAxis dataKey="n" angle={-30} textAnchor="end" interval={0} fontSize={10} height={80} /><YAxis fontSize={10} /><Tooltip />
                <Bar isAnimationActive={false} dataKey={k}>{cb.map((b) => <Cell key={b.n} fill={COLORS[b.n]} />)}<ErrorBar dataKey={e} width={3} stroke="#333" /></Bar></BarChart></ResponsiveContainer></div></div>
          ))}
        </div>
        <table><thead><tr><th>Method</th><th>precision@10</th><th>recall@10</th><th>NDCG@10</th><th>user-side P@10</th><th>user-side lift</th></tr></thead><tbody>
          {cn.map((n) => <tr key={n}><td><span className="dot" style={{ display: "inline-block", background: COLORS[n], marginRight: 6 }} />{n}</td>
            <td>{pm(r.cold_main[n].p)}</td><td>{pm(r.cold_main[n].r)}</td><td>{pm(r.cold_main[n].ndcg)}</td><td>{pm(r.cold_main[n].user_p)}</td><td>{pm(r.cold_main[n].user_lift, 2)}</td></tr>)}
        </tbody></table>
        <p className="note">Collaborative methods have no information about these films, so they perform at chance. The hybrid reaches them through content edges.</p>
      </div>
      <div className="card">
        <h3>Popularity — do the methods recommend mostly popular films? (Q3)</h3>
        <p className="sub">Share of recommendations outside the top-10% most-rated films, and catalogue coverage (fraction of films recommended to at least one user). Accurate methods stay almost entirely in the head of the distribution.</p>
        <div className="grid g2">
          {[["tail", "share outside top-10% popular"], ["cov", "catalogue coverage"]].map(([k, t]) => (
            <div key={k}><div className="note"><b>{t}</b></div><div style={{ height: 270 }}><ResponsiveContainer>
              <BarChart data={pb} margin={{ bottom: 70 }}><CartesianGrid strokeDasharray="3 3" opacity={0.3} /><XAxis dataKey="n" angle={-30} textAnchor="end" interval={0} fontSize={10} height={80} /><YAxis fontSize={10} /><Tooltip />
                <Bar isAnimationActive={false} dataKey={k}>{pb.map((b) => <Cell key={b.n} fill={COLORS[b.n]} />)}</Bar></BarChart></ResponsiveContainer></div></div>
          ))}
        </div>
      </div>
    </>
  );
}
