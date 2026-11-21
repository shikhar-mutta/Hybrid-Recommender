import { useEffect, useState } from "react";

export const COLORS = {
  "Popularity": "#9e9e9e", "Item-based CF": "#1f77b4", "Content only (profile)": "#2ca02c", "Content only": "#2ca02c",
  "RWR (ratings only)": "#ff7f0e", "RWR + popularity discount": "#d4a017", "Late fusion (CF+content)": "#8c564b",
  "Hybrid RWR": "#d62728", "Hybrid RWR + popularity discount": "#9467bd", "Hybrid + popularity discount": "#9467bd",
  "node2vec on hybrid graph": "#17becf", "node2vec": "#17becf", "Random": "#bdbdbd",
};
export const fmt = (v, d = 3) => (v == null ? "–" : Number(v).toFixed(d));
export const pm = (a, d = 3) => `${fmt(a[0], d)} ± ${fmt(a[1], d)}`;

const FILES = ["results", "films", "users", "communities", "vocab"];
export function useData() {
  const [d, setD] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => {
    Promise.all(FILES.map((f) => fetch(`/data/${f}.json`).then((r) => { if (!r.ok) throw new Error(f); return r.json(); })))
      .then((a) => setD(Object.fromEntries(FILES.map((f, i) => [f, a[i]]))))
      .catch((e) => setErr(String(e)));
  }, []);
  return { d, err };
}

export function useApi() {
  const [up, setUp] = useState(false);
  useEffect(() => {
    const ping = () => fetch("/api/health").then((r) => r.ok ? r.json() : Promise.reject()).then(() => setUp(true)).catch(() => setUp(false));
    ping(); const t = setInterval(ping, 5000); return () => clearInterval(t);
  }, []);
  return up;
}
