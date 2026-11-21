import React, { useEffect, useMemo, useRef, useState } from "react";

// Searchable multi-select dropdown with checkboxes (only values from `options` can be chosen).
export default function MultiSelect({ label, options, value, onChange, placeholder = "Select…", max = 99, allowCustom = false }) {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const ref = useRef();
  useEffect(() => {
    const h = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, []);
  const shown = useMemo(() => {
    const s = q.trim().toLowerCase();
    return options.filter((o) => !s || o.toLowerCase().includes(s)).slice(0, 300);
  }, [options, q]);
  const typed = q.trim().replace(/,/g, " ").replace(/\s+/g, " ").toLowerCase();
  const canAdd = allowCustom && typed.length >= 2 && typed.length <= 40 && !options.includes(typed) && !value.includes(typed) && value.length < max;
  const addCustom = () => { if (canAdd) { onChange([...value, typed]); setQ(""); } };
  const toggle = (o) => {
    if (value.includes(o)) onChange(value.filter((x) => x !== o));
    else if (value.length < max) onChange([...value, o]);
  };
  return (
    <div className="ms" ref={ref}>
      <div className="note" style={{ marginBottom: 2 }}>{label}{max < 99 ? ` (max ${max})` : ""}</div>
      <div className="ms-box" onClick={() => setOpen(true)}>
        {value.length === 0 && <span className="note">{placeholder}</span>}
        {value.map((v) => (
          <span key={v} className="chip ms-chip">{v}<b onClick={(e) => { e.stopPropagation(); toggle(v); }} title="remove"> ×</b></span>
        ))}
        <span className="ms-caret">▾</span>
      </div>
      {open && (
        <div className="ms-pop">
          <input autoFocus placeholder={allowCustom ? `Search ${options.length} tags or type a new one…` : `Search ${options.length} options…`} value={q}
            onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); canAdd ? addCustom() : shown.length === 1 && !value.includes(shown[0]) && toggle(shown[0]); } }} />
          <div className="ms-actions">
            <span className="note">{value.length} selected</span>
            <a onClick={() => onChange([])}>clear</a>
            <a onClick={() => setOpen(false)}>done</a>
          </div>
          <div className="ms-list">
            {canAdd && <div className="ms-opt ms-add" onClick={addCustom}>＋ Add “{typed}” as a new tag <span className="note">(Enter)</span></div>}
            {shown.length === 0 && !canAdd && <div className="note" style={{ padding: 8 }}>{allowCustom ? "No match." : "No match — only listed values can be used."}</div>}
            {shown.map((o) => {
              const on = value.includes(o), dis = !on && value.length >= max;
              return (
                <label key={o} className={"ms-opt" + (dis ? " dis" : "")}>
                  <input type="checkbox" checked={on} disabled={dis} onChange={() => toggle(o)} /> {o}
                </label>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
