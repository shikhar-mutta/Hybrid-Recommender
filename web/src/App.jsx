import React, { useState } from "react";
import { useData, useApi } from "./shared.js";
import Overview from "./pages/Overview.jsx";
import Recommend from "./pages/Recommend.jsx";
import NewFilm from "./pages/NewFilm.jsx";
import Results from "./pages/Results.jsx";
import Cold from "./pages/Cold.jsx";
import Communities from "./pages/Communities.jsx";
import Report from "./pages/Report.jsx";

const TABS = [["Overview", Overview], ["Recommendations", Recommend], ["New film (cold start)", NewFilm],
  ["Accuracy & λ", Results], ["Cold start & popularity", Cold], ["Communities", Communities], ["Report & figures", Report]];

class Boundary extends React.Component {
  constructor(p) { super(p); this.state = { e: null }; }
  static getDerivedStateFromError(e) { return { e }; }
  render() { return this.state.e ? <div className="warn">This page crashed: {String(this.state.e)}</div> : this.props.children; }
}

export default function App() {
  const [tab, setTabRaw] = useState(() => Math.min(6, +new URLSearchParams(location.hash.slice(1)).get("tab") || 0));
  const setTab = (i) => { setTabRaw(i); history.replaceState(null, "", "#tab=" + i); };
  const { d, err } = useData();
  const api = useApi();
  const Page = TABS[tab][1];
  return (
    <>
      <header>
        <h1>Hybrid Recommender — users · films · content</h1>
        <span>AIM608 End-term Project 4 · MovieLens latest-small</span>
        <span style={{ marginLeft: "auto" }} title="Live API: python3 server/api.py">
          <span className="dot" style={{ display: "inline-block", background: api ? "#2ca02c" : "#bbb", marginRight: 6 }} />
          live API {api ? "connected" : "offline"}
        </span>
      </header>
      <nav>{TABS.map(([n], i) => <button key={n} className={i === tab ? "on" : ""} onClick={() => setTab(i)}>{n}</button>)}</nav>
      <main>
        {err && <div className="warn">Could not load data ({err}). Run <code>python3 build/export_web_data.py</code> first.</div>}
        {!d && !err && <p>Loading…</p>}
        {d && <Boundary key={tab}><Page d={d} api={api} go={setTab} /></Boundary>}
      </main>
    </>
  );
}
