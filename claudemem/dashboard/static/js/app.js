/* =============================================================================
   claude-memory hub — front-end logic.
   One Alpine component drives the whole SPA. Vendored libs (echarts, cytoscape,
   markdown-it) are optional: each panel degrades gracefully if its lib is absent.
   Talks only to the existing JSON API in dashboard/api.py.
   ============================================================================= */
"use strict";

const PHI = 1.618;
const API = (p) => fetch(p).then((r) => (r.ok ? r.json() : Promise.reject(r.status)));
const POST = (p, body) =>
  fetch(p, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) })
    .then((r) => (r.ok ? r.json() : Promise.reject(r.status)));

/* small inline SVG icon set (stroke=currentColor) ------------------------- */
const ICONS = {
  search: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3" stroke-linecap="round"/></svg>',
  notes: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 4h11l5 5v11H4z"/><path d="M14 4v6h6" opacity=".6"/><path d="M8 13h8M8 17h6" stroke-linecap="round"/></svg>',
  graph: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="6" cy="6" r="2.4"/><circle cx="18" cy="7" r="2.4"/><circle cx="12" cy="18" r="2.4"/><path d="M7.8 7.4l2.6 8M16.2 8.6l-2.6 7.4M8 6.4l8 .4" opacity=".7"/></svg>',
  metrics: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 4v16h16"/><path d="M8 14l3-4 3 2 4-6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  injections: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2L4 14h7l-1 8 9-12h-7z" stroke-linejoin="round"/></svg>',
  promotions: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3l2.5 5 5.5.8-4 4 1 5.5L12 16l-5 2.3 1-5.5-4-4 5.5-.8z" stroke-linejoin="round"/></svg>',
  settings: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19 12a7 7 0 0 0-.1-1l2-1.5-2-3.5-2.4 1a7 7 0 0 0-1.7-1l-.4-2.5h-4l-.4 2.5a7 7 0 0 0-1.7 1l-2.4-1-2 3.5 2 1.5a7 7 0 0 0 0 2l-2 1.5 2 3.5 2.4-1a7 7 0 0 0 1.7 1l.4 2.5h4l.4-2.5a7 7 0 0 0 1.7-1l2.4 1 2-3.5-2-1.5a7 7 0 0 0 .1-1z"/></svg>',
};

const TYPE_COLORS = {
  feedback: "#f0b35e", project: "#5ec8d8", reference: "#9d8cff",
  user: "#5ed8a0", missing: "#5b657a", entity: "#e9c46a",
};

function hub() {
  return {
    /* ---- state ---- */
    nav: [
      { id: "search", label: "Search", key: "1" },
      { id: "notes", label: "Notes", key: "2" },
      { id: "graph", label: "Graph", key: "3" },
      { id: "metrics", label: "Metrics", key: "4" },
      { id: "injections", label: "Injections", key: "5" },
      { id: "promotions", label: "Promotions", key: "6" },
      { id: "settings", label: "Settings", key: "7" },
    ],
    icons: ICONS,
    view: "search",
    stats: {},
    toasts: [],

    /* search */
    q: "", kind: "all", rerank: true, res: [], facts: [], timing: "", searched: false,
    loading: false, _debounce: null, _seq: 0,
    suggestions: ["deploy", "golden ratio", "commit", "never delete", "analytics"],

    /* notes */
    allNotes: [], noteType: null, note: null,

    /* graph */
    graph: {}, graphFilter: [], _cy: null, hasCyto: false,

    /* metrics */
    metrics: {}, hasECharts: false, _charts: {},

    /* injections */
    injections: [], _injSeen: new Set(), sseInjLive: false, _injSse: null,

    /* promotions */
    promotions: [],

    /* settings / index */
    indexRunning: false, indexLog: [], _idxSse: null,

    /* markdown */
    _md: null,

    /* =================== lifecycle =================== */
    init() {
      this.hasCyto = typeof window.cytoscape === "function";
      this.hasECharts = typeof window.echarts === "object";
      if (window.markdownit) {
        try { this._md = window.markdownit({ html: false, linkify: true, typographer: true }); } catch (e) {}
      }
      this.loadStats();
      // restore deep-link view from hash
      const h = (location.hash || "").replace("#", "");
      if (this.nav.some((n) => n.id === h)) this.view = h;
      this.enter(this.view);

      // keyboard nav: 1..7 and "/" focus search
      window.addEventListener("keydown", (e) => {
        if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") {
          if (e.key === "Escape") e.target.blur();
          return;
        }
        if (e.key === "/") { e.preventDefault(); this.go("search"); this.$nextTick(() => document.querySelector(".searchbar input")?.focus()); return; }
        const n = this.nav.find((x) => x.key === e.key);
        if (n) this.go(n.id);
      });
      // refresh stats periodically (cheap)
      setInterval(() => this.loadStats(), 15000);
    },

    go(id) {
      if (this.view === id) return;
      this.view = id;
      history.replaceState(null, "", "#" + id);
      this.enter(id);
    },

    // lazy-load per-section data the first time it's shown
    enter(id) {
      if (id === "notes" && !this.allNotes.length) this.loadNotes();
      if (id === "graph") this.loadGraph();
      if (id === "metrics") this.loadMetrics();
      if (id === "injections") this.startInjections();
      if (id === "promotions") this.loadPromotions();
      if (id === "settings") this.loadStats();
    },

    subtitle() {
      switch (this.view) {
        case "search": return "hybrid recall across all sessions";
        case "notes": return "your curated knowledge";
        case "graph": return "wikilink relationships";
        case "metrics": return "corpus observability";
        case "injections": return "audit window";
        case "promotions": return "candidate notes";
        case "settings": return "health & control";
      }
      return "";
    },

    refreshAll() {
      this.loadStats();
      this.enter(this.view);
      this.toast("refreshed", "ok");
    },

    /* =================== helpers =================== */
    fmt(n) { return n == null ? "—" : Number(n).toLocaleString(); },
    scorePct(s) { return Math.max(6, Math.min(100, Math.round((s / 6) * 100))); },
    embedPct() {
      const c = this.stats.chunks, e = this.stats.chunks_embedded;
      if (!c) return "0"; return Math.round((e / c) * 100).toString();
    },
    typeColor(t) { return TYPE_COLORS[t] || "#8b95ab"; },
    shortPath(p) {
      if (!p) return ""; const m = p.replace(/\\/g, "/").split("/"); return m.slice(-2).join("/");
    },
    shortTime(ts) {
      if (!ts) return "—";
      const d = new Date(ts.replace(" ", "T"));
      if (isNaN(d)) return ts;
      return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    },
    esc(s) { return (s || "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); },
    highlight(text) {
      const safe = this.esc(text);
      const terms = (this.q || "").split(/\s+/).filter((t) => t.length > 2).slice(0, 6);
      if (!terms.length) return safe;
      const re = new RegExp("(" + terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|") + ")", "gi");
      return safe.replace(re, "<mark>$1</mark>");
    },
    renderMd(text) {
      if (!text) return "";
      if (this._md) return this._md.render(text);
      return "<p>" + this.esc(text).replace(/\n/g, "<br>") + "</p>";
    },
    toast(msg, cls = "") {
      const id = ++this._seq;
      this.toasts.push({ id, msg, cls });
      setTimeout(() => { this.toasts = this.toasts.filter((t) => t.id !== id); }, 3200);
    },

    /* =================== stats =================== */
    async loadStats() {
      try { this.stats = await API("/api/stats"); } catch (e) { /* keep last */ }
    },

    /* =================== search =================== */
    onType() {
      clearTimeout(this._debounce);
      if (!this.q.trim()) { this.searched = false; this.res = []; this.facts = []; return; }
      this.loading = true; this.searched = true;
      this._debounce = setTimeout(() => this.doSearch(), 250);
    },
    async doSearch() {
      const q = this.q.trim();
      if (!q) { this.searched = false; this.res = []; this.facts = []; this.loading = false; return; }
      this.searched = true; this.loading = true;
      const my = ++this._seq;
      try {
        const d = await API(`/api/search?q=${encodeURIComponent(q)}&kind=${this.kind}&k=10&rerank=${this.rerank}`);
        if (my !== this._seq) return; // stale response
        this.res = d.results || [];
        this.facts = d.facts || [];
        this.timing = `${d.timing_ms} ms · ${this.res.length} recalls · ${this.facts.length} notes`;
      } catch (e) {
        this.toast("search failed (" + e + ")", "err");
      } finally {
        if (my === this._seq) this.loading = false;
      }
    },

    /* =================== notes =================== */
    get noteTypes() { return [...new Set(this.allNotes.map((n) => n.type))].sort(); },
    async loadNotes() {
      try { const d = await API("/api/facts"); this.allNotes = d.facts || []; }
      catch (e) { this.toast("could not load notes", "err"); }
    },
    groupedNotes() {
      const list = this.noteType ? this.allNotes.filter((n) => n.type === this.noteType) : this.allNotes;
      const by = {};
      for (const n of list) (by[n.project] = by[n.project] || []).push(n);
      return Object.keys(by).sort().map((project) => ({
        project,
        notes: by[project].sort((a, b) => a.title.localeCompare(b.title)),
      }));
    },
    async openNote(id) {
      this.go("notes");
      try {
        const f = await API(`/api/fact/${id}`);
        if (f.error) { this.toast("note not found", "err"); return; }
        f._backlinks = this.backlinksFor(f);
        this.note = f;
      } catch (e) { this.toast("could not open note", "err"); }
    },
    // derive related notes from [[wikilinks]] in the body, resolving to ids when possible
    backlinksFor(f) {
      const links = [];
      const re = /\[\[([^\]]+)\]\]/g; let m; const seen = new Set();
      const body = f.body || "";
      while ((m = re.exec(body))) {
        const label = m[1].split("|")[0].trim();
        if (seen.has(label.toLowerCase())) continue; seen.add(label.toLowerCase());
        const hit = this.allNotes.find(
          (n) => n.title.toLowerCase() === label.toLowerCase() ||
                 (n.path || "").toLowerCase().includes("/" + label.toLowerCase() + ".md")
        );
        links.push({ id: hit ? hit.id : -1, label });
      }
      return links;
    },

    /* =================== graph =================== */
    get graphTypes() {
      if (!this.graph.nodes) return [];
      return [...new Set(this.graph.nodes.map((n) => n.type))].sort();
    },
    toggleGraphType(t) {
      this.graphFilter = this.graphFilter.includes(t)
        ? this.graphFilter.filter((x) => x !== t)
        : [...this.graphFilter, t];
      this.renderGraph();
    },
    async loadGraph() {
      if (!this.hasCyto) return;
      if (!this.graph.nodes) {
        try { this.graph = await API("/api/graph"); this.graphFilter = this.graphTypes.slice(); }
        catch (e) { this.toast("could not load graph", "err"); return; }
      }
      this.$nextTick(() => this.renderGraph());
    },
    renderGraph() {
      if (!this.hasCyto || !this.graph.nodes) return;
      const el = document.getElementById("cy");
      if (!el) return;
      const keep = new Set(this.graph.nodes.filter((n) => this.graphFilter.includes(n.type)).map((n) => n.id));
      const nodes = this.graph.nodes
        .filter((n) => keep.has(n.id))
        .map((n) => ({ data: { id: n.id, label: n.label, type: n.type, group: n.group } }));
      const edges = this.graph.edges
        .filter((e) => keep.has(e.source) && keep.has(e.target))
        .map((e, i) => ({ data: { id: "e" + i, source: e.source, target: e.target, kind: e.kind } }));

      if (this._cy) { this._cy.destroy(); this._cy = null; }
      this._cy = window.cytoscape({
        container: el,
        elements: { nodes, edges },
        wheelSensitivity: 0.22,
        style: [
          { selector: "node", style: {
              "background-color": (n) => this.typeColor(n.data("type")),
              "label": "data(label)", "color": "#c3cbdb", "font-size": 9,
              "font-family": "Inter, sans-serif", "text-wrap": "wrap", "text-max-width": 110,
              "text-valign": "bottom", "text-margin-y": 5,
              "width": 18, "height": 18, "border-width": 2, "border-color": "#0e1117",
              "text-outline-color": "#0a0c10", "text-outline-width": 2,
              "transition-property": "width height border-color", "transition-duration": "120ms",
            } },
          { selector: "node[type='missing']", style: { "background-opacity": 0.4, "border-style": "dashed", "border-color": "#3a4256" } },
          { selector: "node:selected", style: { "border-color": "#e9c46a", "border-width": 3, "width": 24, "height": 24 } },
          { selector: "edge", style: {
              "width": 1.2, "line-color": "#2a3346", "curve-style": "bezier",
              "target-arrow-color": "#2a3346", "target-arrow-shape": "triangle", "arrow-scale": 0.7,
              "opacity": 0.7,
            } },
          { selector: "node.dim", style: { "opacity": 0.18 } },
          { selector: "edge.dim", style: { "opacity": 0.06 } },
        ],
        layout: { name: "cose", animate: false, padding: 40, nodeRepulsion: 9000, idealEdgeLength: 90, gravity: 0.3 },
      });

      // click a real note node → open it; hover → highlight neighborhood
      this._cy.on("tap", "node", (ev) => {
        const id = ev.target.id();
        const hit = this.allNotes.find((n) => n.title === id || (n.path || "").replace(/\\/g, "/").toLowerCase().endsWith("/" + id.toLowerCase() + ".md"));
        if (hit) this.openNote(hit.id);
        else { this.loadNotes().then(() => { const h = this.allNotes.find((n) => n.title === id); if (h) this.openNote(h.id); else this.toast("unresolved link node", ""); }); }
      });
      this._cy.on("mouseover", "node", (ev) => {
        const nbh = ev.target.closedNeighborhood();
        this._cy.elements().addClass("dim");
        nbh.removeClass("dim");
      });
      this._cy.on("mouseout", "node", () => this._cy.elements().removeClass("dim"));
    },

    /* =================== metrics =================== */
    async loadMetrics() {
      if (!this.hasECharts) { this.loadStats(); return; }
      try { this.metrics = await API("/api/metrics"); } catch (e) { return; }
      await this.loadStats();
      this.$nextTick(() => this.renderCharts());
    },
    renderCharts() {
      if (!this.hasECharts) return;
      const grid = { left: 48, right: 24, top: 24, bottom: 36 };
      const axisCommon = {
        axisLine: { lineStyle: { color: "#2a3346" } },
        axisLabel: { color: "#8b95ab", fontSize: 11 },
        splitLine: { lineStyle: { color: "#161c28" } },
      };
      // growth: chunks / facts / sources over their recorded timestamps
      const series = [];
      const mk = (key, name, color) => {
        const rows = (this.metrics[key] || []).map((r) => [r.ts, r.value]);
        return { name, type: "line", smooth: true, showSymbol: rows.length < 12, data: rows,
          lineStyle: { width: 2, color }, itemStyle: { color },
          areaStyle: { color: { type: "linear", x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [{ offset: 0, color: color + "44" }, { offset: 1, color: color + "00" }] } } };
      };
      series.push(mk("chunks", "chunks", "#e9c46a"));
      series.push(mk("facts", "notes", "#5ec8d8"));
      series.push(mk("sources", "sources", "#9d8cff"));

      this._chart("chart-growth", {
        backgroundColor: "transparent",
        tooltip: { trigger: "axis", backgroundColor: "#141925", borderColor: "#222a3a", textStyle: { color: "#f3f5fa" } },
        legend: { textStyle: { color: "#8b95ab" }, top: 0, right: 0, icon: "roundRect" },
        grid, xAxis: { type: "time", ...axisCommon }, yAxis: { type: "value", ...axisCommon }, series,
      });

      // embedding coverage gauge-ish bar (embedded vs total)
      const total = this.stats.chunks || 0, emb = this.stats.chunks_embedded || 0;
      this._chart("chart-embed", {
        backgroundColor: "transparent",
        tooltip: { trigger: "axis", backgroundColor: "#141925", borderColor: "#222a3a", textStyle: { color: "#f3f5fa" } },
        grid: { left: 90, right: 24, top: 12, bottom: 24 },
        xAxis: { type: "value", max: total || 1, ...axisCommon },
        yAxis: { type: "category", data: ["embedded", "indexed"], ...axisCommon },
        series: [{
          type: "bar", barWidth: 22,
          data: [
            { value: emb, itemStyle: { color: "#5ed8a0", borderRadius: 6 } },
            { value: total, itemStyle: { color: "#3a4256", borderRadius: 6 } },
          ],
          label: { show: true, position: "right", color: "#c3cbdb", formatter: (p) => p.value.toLocaleString() },
        }],
      });
    },
    _chart(id, opt) {
      const el = document.getElementById(id);
      if (!el) return;
      let c = this._charts[id];
      if (!c) { c = window.echarts.init(el, null, { renderer: "canvas" }); this._charts[id] = c;
        window.addEventListener("resize", () => c.resize()); }
      c.setOption(opt, true);
    },

    /* =================== injections (live audit) =================== */
    async startInjections() {
      // initial backfill
      try {
        const d = await API("/api/injections?limit=50");
        this.injections = (d.injections || []);
        this.injections.forEach((i) => this._injSeen.add(i.id));
      } catch (e) {}
      // live SSE tail (api emits the newest row when it changes)
      if (this._injSse || typeof EventSource === "undefined") return;
      try {
        const es = new EventSource("/api/injections/stream");
        this._injSse = es;
        es.addEventListener("injection", (ev) => {
          this.sseInjLive = true;
          try {
            const row = JSON.parse(ev.data);
            if (this._injSeen.has(row.id)) return;
            this._injSeen.add(row.id);
            row._fresh = true;
            this.injections.unshift(row);
            if (this.injections.length > 200) this.injections.pop();
            setTimeout(() => { row._fresh = false; }, 1500);
            this.loadStats();
          } catch (e) {}
        });
        es.onopen = () => { this.sseInjLive = true; };
        es.onerror = () => { this.sseInjLive = false; };
      } catch (e) {}
    },

    /* =================== promotions =================== */
    async loadPromotions() {
      try { const d = await API("/api/promotions"); this.promotions = (d.promotions || []).map((p) => ({ ...p, _busy: false })); }
      catch (e) { this.toast("could not load promotions", "err"); }
    },
    supportLabel(p) {
      const s = p.support || {};
      const bits = [];
      if (p.score != null) bits.push("score " + Number(p.score).toFixed(2));
      if (s.count != null) bits.push(s.count + " occurrences");
      if (s.sessions) bits.push((Array.isArray(s.sessions) ? s.sessions.length : s.sessions) + " sessions");
      if (p.status) bits.push(p.status);
      return bits.join(" · ");
    },
    async actPromo(id, action) {
      const p = this.promotions.find((x) => x.id === id);
      if (p) p._busy = true;
      try {
        await POST(`/api/promotions/${id}`, { action });
        this.promotions = this.promotions.filter((x) => x.id !== id);
        this.toast(action === "accept" ? "note promoted & indexed" : "candidate rejected", action === "accept" ? "ok" : "");
        if (action === "accept") { this.allNotes = []; this.loadStats(); }
      } catch (e) {
        if (p) p._busy = false;
        this.toast("action failed (" + e + ")", "err");
      }
    },

    /* =================== settings: kill switch + index =================== */
    async toggleKill(on) {
      try {
        const d = await POST("/api/killswitch", { on });
        this.stats = { ...this.stats, killswitch: d.on };
        this.toast(d.on ? "memory DISABLED" : "memory re-enabled", d.on ? "err" : "ok");
      } catch (e) { this.toast("kill switch failed", "err"); this.loadStats(); }
    },
    async runIndex(full) {
      try {
        const d = await POST("/api/index", { full });
        if (!d.started) { this.toast("index already running", ""); }
        else { this.indexRunning = true; this.indexLog = []; this.toast(full ? "full reindex started" : "reindex started", "ok"); this.streamIndex(); }
      } catch (e) { this.toast("could not start index", "err"); }
    },
    streamIndex() {
      if (typeof EventSource === "undefined") return;
      if (this._idxSse) { this._idxSse.close(); this._idxSse = null; }
      const es = new EventSource("/api/index/stream");
      this._idxSse = es;
      es.addEventListener("progress", (ev) => {
        const text = ev.data;
        const cls = /error/i.test(text) ? "err" : (/complete/i.test(text) ? "done" : "");
        this.indexLog.push({ text, cls });
        if (this.indexLog.length > 400) this.indexLog.shift();
        this.$nextTick(() => { const el = this.$refs.indexlog; if (el) el.scrollTop = el.scrollHeight; });
        if (cls === "done") { this.indexRunning = false; this.loadStats(); }
      });
      es.addEventListener("done", () => {
        // server reports idle once no progress is pending
        if (this.indexRunning) { this.indexRunning = false; this.loadStats(); this.toast("index complete", "ok"); }
        es.close(); this._idxSse = null;
      });
      es.onerror = () => { /* keep last log; allow reconnect on next run */ };
    },
  };
}

// expose for Alpine x-data="hub()"
window.hub = hub;
