import React, { useEffect, useRef, useState } from 'react';
import embed from 'vega-embed';


export default function App() {
const [query, setQuery] = useState('');
const [loading, setLoading] = useState(false);
const [error, setError] = useState(null);
const visRef = useRef(null);


const run = async () => {
setError(null);
if (!query.trim()) return setError('Please type a query');
setLoading(true);
try {
const res = await fetch('http://localhost:8000/query', {
method: 'POST',
headers: { 'Content-Type': 'application/json' },
body: JSON.stringify({ query }),
});
const data = await res.json();
if (!res.ok) throw new Error(data.detail || 'Server error');
 let spec = data.spec || {};
 // ensure responsive, larger default size when backend doesn't specify
 if (!spec.width) spec.width = 'container';
 if (!spec.height) spec.height = 500;
 if (!spec.autosize) spec.autosize = { type: 'fit', contains: 'padding' };
 // render vega-lite spec
if (visRef.current) {
  visRef.current.innerHTML = '';
}
 embed(visRef.current || '#vis', spec, { actions: false }).catch(err => console.error(err));
} catch (e) {
setError(e.message);
} finally {
setLoading(false);
}
};


return (
<div className="container">
  <div className="header">
    <div>
      <div className="title">TalkToADatabase</div>
      <div className="subtitle">Ask questions about your data and get instant charts.</div>
    </div>
  </div>

  <div className="panel">
    <div className="stack">
      <textarea
        className="textarea"
        placeholder="e.g. Top 10 players by overall rating in 2019"
        rows={4}
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <div className="row">
        <button className="btn" onClick={run} disabled={loading}>{loading ? 'Running…' : 'Run query'}</button>
        <div className="hint">Press Enter + Run to visualize. Vega-Lite renders below.</div>
      </div>
      {error && <div className="error">{error}</div>}
    </div>
  </div>

  <div className="card">
    <div ref={visRef} id="vis"></div>
  </div>
</div>
);
}