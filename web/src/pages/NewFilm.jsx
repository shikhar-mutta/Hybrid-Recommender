import React, { useState } from "react";

const PRESETS = {
  "Toy Story 4 (2019)": { title: "Toy Story 4", genres: "Adventure, Animation, Children, Comedy, Fantasy", tags: "pixar, toys, fun" },
  "John Wick: Chapter 3 (2019)": { title: "John Wick Chapter 3 Parabellum", genres: "Action, Crime, Thriller", tags: "assassin, hitman, gun fu, keanu reeves" },
  "Dangal (2016)": { title: "Dangal", genres: "Action, Drama", tags: "wrestling, sports, father daughter, india, inspirational, bollywood" },
};

export default function NewFilm({ api }) {
  const [f, setF] = useState({ ...PRESETS["Toy Story 4 (2019)"], lam: 0.1 });
  const [out, setOut] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const go = () => {
    setBusy(true); setErr(null);
    fetch("/api/newfilm", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(f) })
      .then((r) => r.json()).then((j) => (j.error ? setErr(j.error) : setOut(j))).catch((e) => setErr(String(e))).finally(() => setBusy(false));
  };
  return (
    <>
      <div className="card">
        <h3>Add a film that nobody has rated — which users should get it?</h3>
        <p className="sub">The film has no rating edge, so collaborative filtering gives it a score of exactly 0 for every user. The hybrid embeds its text, links it to its 20 most similar films and runs the walk from every user.
          Users are ranked by the film’s percentile on their shelf of <i>new releases</i> (films from 2015 on they have not seen): 1% = first on that shelf.</p>
        {!api && <div className="warn" style={{ marginBottom: 10 }}>Live API is offline. Start it with <code>python3 server/api.py</code> in the project folder (first request loads the sentence-transformer model).</div>}
        <div className="row" style={{ marginBottom: 10 }}>
          {Object.keys(PRESETS).map((k) => <button key={k} className="btn sec" onClick={() => setF({ ...PRESETS[k], lam: f.lam })}>{k}</button>)}
        </div>
        <div className="grid g3">
          <label className="s">Title<input value={f.title} onChange={(e) => setF({ ...f, title: e.target.value })} /></label>
          <label className="s">Genres (comma separated)<input value={f.genres} onChange={(e) => setF({ ...f, genres: e.target.value })} /></label>
          <label className="s">Tags<input value={f.tags} onChange={(e) => setF({ ...f, tags: e.target.value })} /></label>
        </div>
        <div className="row" style={{ marginTop: 10 }}>
          <label className="s">λ content weight: <b>{f.lam}</b><input type="range" min="0.01" max="0.5" step="0.01" value={f.lam} onChange={(e) => setF({ ...f, lam: +e.target.value })} /></label>
          <button className="btn" onClick={go} disabled={!api || busy}>{busy ? "Running…" : "Recommend to users"}</button>
        </div>
        {err && <div className="warn" style={{ marginTop: 10 }}>{err}</div>}
      </div>
      {out && (
        <div className="grid g2">
          <div className="card">
            <h3>Content neighbours it was linked to</h3>
            <p className="sub">Pure CF: score = 0 for all users. Hybrid: reaches {out.reached} of 610 users.</p>
            <table><thead><tr><th>Film</th><th>Genres</th><th>cosine</th></tr></thead><tbody>
              {out.neighbours.map((n) => <tr key={n.title}><td>{n.title}</td><td style={{ textAlign: "left" }}>{n.genres}</td><td>{n.cos}</td></tr>)}
            </tbody></table>
          </div>
          <div className="card">
            <h3>Top users for this film</h3>
            <p className="sub">percentile on the user’s shelf of {out.shelf} new releases</p>
            <table><thead><tr><th>User</th><th>percentile</th><th>their favourites</th></tr></thead><tbody>
              {out.users.map((u) => <tr key={u.user}><td>{u.user}</td><td>{u.percentile}%</td><td style={{ textAlign: "left" }}>{u.fav.join("; ")}</td></tr>)}
            </tbody></table>
          </div>
        </div>
      )}
    </>
  );
}
