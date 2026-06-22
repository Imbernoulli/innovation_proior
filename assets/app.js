/* Innovation Prior — data-driven single-page app.
 * Four browse modes:
 *  - Methods:      atomic reasoning traces. methods.json -> #<slug>/<tab>,
 *                  fetches methods/<slug>/results/<tab>.md (Context/Reasoning/Answer).
 *  - Trajectories: iterative research lines over an MLS-Bench task. trajectories.json
 *                  -> #t/<task>, reads trajectories/<task>/meta.json then walks its
 *                  ordered files (initial context -> baselines -> feedback -> finale).
 *  - Agentic:      the SAME research lines rendered as an *agent transcript* — the model
 *                  edits code and calls run_experiment via tool calls. agentic.json ->
 *                  #a/<task>, reads trajectories/<task>/agentic_messages.json (a list of
 *                  system/user/assistant(+reasoning,tool_calls)/tool messages).
 *  - Training:     the full SFT corpus the traces are assembled into. sft/viewer/index.json
 *                  lists every example; bodies lazy-load (and gunzip in-browser via pako)
 *                  from sft/viewer/<dataset>-NNN.json.gz. #d / #d/<id>.
 * Rendering: markdown (marked) -> math (KaTeX) -> code (highlight.js).
 */
(function () {
  "use strict";

  var TABS = ["context", "reasoning", "answer"];
  var DEFAULT_TAB = "reasoning";

  var methods = [];           // [{slug,title,domain,arxiv}]
  var bySlug = {};            // slug -> method
  var trajectories = [];      // [{task,title,domain,endpoint}]
  var trajByTask = {};        // task -> trajectory
  var agentics = [];          // [{task,title,domain,year,n_steps,n_actions,...}]
  var agByTask = {};          // task -> agentic index entry
  var trainIndex = null;      // {total,datasets,examples:[...]} (lazy)
  var trainById = {};         // id -> example index entry
  var trainIndexPromise = null;
  var shardCache = {};        // shard name -> [examples]
  var agCache = {};           // task -> transcript record

  var collapsed = {};         // group name -> bool
  var MODE = "methods";       // methods | trajectories | agentic | training
  var builtMode = null;       // which mode the sidebar DOM currently reflects
  var current = { slug: null, tab: DEFAULT_TAB, task: null, agtask: null, id: null };
  var fetchToken = 0;         // guards against out-of-order responses
  var mdCache = {};           // "slug/tab" -> text

  // ---- DOM refs ----
  var $ = function (id) { return document.getElementById(id); };
  var nav = $("nav");
  var searchInput = $("search");
  var hero = $("hero");
  var article = $("article");
  var contentEl = $("content");
  var traj = $("traj");
  var trajContentEl = $("traj-content");
  var agentic = $("agentic");
  var agContentEl = $("ag-content");
  var train = $("train");
  var trainContentEl = $("train-content");
  var sidebar = $("sidebar");
  var scrim = $("scrim");
  var menuToggle = $("menu-toggle");
  var modeBtns = {
    methods: $("mode-methods"),
    trajectories: $("mode-traj"),
    agentic: $("mode-agentic"),
    training: $("mode-train")
  };

  // ---------- markdown / math / code rendering ----------
  function configureMarked() {
    if (typeof marked === "undefined") return;
    marked.setOptions({ gfm: true, breaks: false, headerIds: true, mangle: false });
  }

  function parseMd(text) {
    if (typeof marked === "undefined") return "<pre>" + escapeHtml(text) + "</pre>";
    return marked.parse ? marked.parse(text) : marked(text);
  }

  function renderMath(el) {
    if (typeof renderMathInElement === "undefined") return;
    try {
      renderMathInElement(el, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "\\[", right: "\\]", display: true },
          { left: "\\(", right: "\\)", display: false },
          { left: "$", right: "$", display: false }
        ],
        throwOnError: false,
        ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"]
      });
    } catch (e) { /* non-fatal */ }
  }

  function highlightCode(el) {
    if (typeof hljs === "undefined") return;
    el.querySelectorAll("pre code").forEach(function (block) {
      try { hljs.highlightElement(block); } catch (e) { /* ignore */ }
    });
  }

  function renderInto(el, text) {
    el.innerHTML = parseMd(text);
    renderMath(el);     // order matters: markdown -> math -> code highlight
    highlightCode(el);
    el.scrollTop = 0;
  }

  // Render pre-built HTML (transcripts / training examples) with math + code passes.
  function paintHtml(el, html) {
    el.innerHTML = html;
    renderMath(el);
    highlightCode(el);
    el.scrollTop = 0;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function setStatus(el, kind, msg) {
    el.innerHTML = '<div class="status status-' + kind + '">' + escapeHtml(msg) + "</div>";
  }

  function fetchHint() {
    return (location.protocol === "file:")
      ? " This page must be served over HTTP (e.g. a local server or GitHub Pages); the file:// protocol blocks fetch()."
      : "";
  }

  // ---------- sidebar (mode-generic) ----------
  function trainGroupOf(e) {
    return e.ds === "maintain_sft" ? e.group : ("innovation · " + e.group);
  }

  // Per-mode accessors that drive the sidebar + active-link tracking.
  function navConfig() {
    if (MODE === "trajectories") return {
      list: trajectories, dataAttr: "data-task",
      keyOf: function (m) { return m.task; }, groupOf: function (m) { return m.domain; },
      titleOf: function (m) { return m.title; },
      subOf: function (m) { return m.endpoint ? ("→ " + m.endpoint) : ""; },
      hrefOf: function (m) { return "#t/" + m.task; },
      activeKey: function () { return current.task; },
      searchOf: function (m) { return m.title + " " + m.task + " " + m.domain + " " + (m.endpoint || ""); },
      arxivOf: null
    };
    if (MODE === "agentic") return {
      list: agentics, dataAttr: "data-agtask",
      keyOf: function (m) { return m.task; }, groupOf: function (m) { return m.domain; },
      titleOf: function (m) { return m.title; },
      subOf: function (m) { return m.n_steps + " steps · " + m.n_actions + " tool calls"; },
      hrefOf: function (m) { return "#a/" + m.task; },
      activeKey: function () { return current.agtask; },
      searchOf: function (m) { return m.title + " " + m.task + " " + m.domain + " " + (m.endpoint || ""); },
      arxivOf: null
    };
    if (MODE === "training") return {
      list: (trainIndex ? trainIndex.examples : []), dataAttr: "data-id",
      keyOf: function (m) { return String(m.id); }, groupOf: trainGroupOf,
      titleOf: function (m) { return m.title; }, subOf: function (m) { return m.kind; },
      hrefOf: function (m) { return "#d/" + m.id; },
      activeKey: function () { return current.id; },
      searchOf: function (m) { return m.title + " " + m.kind + " " + m.group + " " + (m.year || ""); },
      arxivOf: null
    };
    return { // methods
      list: methods, dataAttr: "data-slug",
      keyOf: function (m) { return m.slug; }, groupOf: function (m) { return m.domain; },
      titleOf: function (m) { return m.title; }, subOf: function () { return ""; },
      hrefOf: function (m) { return "#" + m.slug + "/" + current.tab; },
      activeKey: function () { return current.slug; },
      searchOf: function (m) { return m.title + " " + m.slug + " " + m.domain; },
      arxivOf: function (m) { return m.arxiv; }
    };
  }

  function groupBy(list, fn) {
    var groups = [], index = {};
    list.forEach(function (m) {
      var key = fn(m);
      if (!(key in index)) { index[key] = groups.length; groups.push({ key: key, items: [] }); }
      groups[index[key]].items.push(m);
    });
    return groups;
  }

  function buildSidebar(filter) {
    var cfg = navConfig();
    var q = (filter || "").trim().toLowerCase();
    var matches = cfg.list.filter(function (m) {
      if (!q) return true;
      return cfg.searchOf(m).toLowerCase().indexOf(q) !== -1;
    });

    nav.innerHTML = "";
    builtMode = MODE;
    if (matches.length === 0) {
      var empty = document.createElement("p");
      empty.className = "nav-empty";
      empty.textContent = "Nothing matches “" + filter + "”.";
      nav.appendChild(empty);
      return;
    }

    var activeKey = cfg.activeKey();

    groupBy(matches, cfg.groupOf).forEach(function (group) {
      var section = document.createElement("div");
      section.className = "nav-group";

      var isCollapsed = !q && collapsed[group.key]; // searching forces expand
      var btn = document.createElement("button");
      btn.className = "nav-group-head";
      btn.type = "button";
      btn.setAttribute("aria-expanded", String(!isCollapsed));
      btn.innerHTML =
        '<span class="caret" aria-hidden="true">▸</span>' +
        '<span class="nav-group-name">' + escapeHtml(group.key) + "</span>" +
        '<span class="nav-group-count">' + group.items.length + "</span>";
      btn.addEventListener("click", function () {
        collapsed[group.key] = !collapsed[group.key];
        buildSidebar(searchInput.value);
      });
      section.appendChild(btn);

      var ul = document.createElement("ul");
      ul.className = "nav-list";
      if (isCollapsed) ul.hidden = true;

      group.items.forEach(function (m) {
        var k = cfg.keyOf(m);
        var li = document.createElement("li");
        var a = document.createElement("a");
        var isActive = k === String(activeKey);
        a.className = "nav-link" + (isActive ? " is-active" : "");
        a.href = cfg.hrefOf(m);
        a.setAttribute(cfg.dataAttr, k);
        var sub = cfg.subOf(m);
        var subHtml = sub ? ('<span class="nav-link-sub">' + escapeHtml(sub) + "</span>") : "";
        a.innerHTML = '<span class="nav-link-title">' + escapeHtml(cfg.titleOf(m)) + "</span>" + subHtml;
        if (isActive) a.setAttribute("aria-current", "page");
        li.appendChild(a);

        var ax = cfg.arxivOf && cfg.arxivOf(m);
        if (ax) {
          var ext = document.createElement("a");
          ext.className = "nav-arxiv";
          ext.href = "https://arxiv.org/abs/" + ax;
          ext.target = "_blank";
          ext.rel = "noopener";
          ext.title = "arXiv:" + ax;
          ext.textContent = ax;
          ext.addEventListener("click", function (e) { e.stopPropagation(); });
          li.appendChild(ext);
        }
        ul.appendChild(li);
      });
      section.appendChild(ul);
      nav.appendChild(section);
    });
  }

  function markActive() {
    var cfg = navConfig();
    var cur = cfg.activeKey();
    nav.querySelectorAll(".nav-link").forEach(function (a) {
      var k = a.getAttribute(cfg.dataAttr);
      var active = k != null && k === String(cur);
      a.classList.toggle("is-active", active);
      if (active) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
      if (MODE === "methods" && a.getAttribute("data-slug")) {
        a.href = "#" + a.getAttribute("data-slug") + "/" + current.tab;
      }
    });
  }

  function updateModeButtons() {
    Object.keys(modeBtns).forEach(function (m) {
      if (modeBtns[m]) modeBtns[m].classList.toggle("is-active", MODE === m);
    });
    searchInput.placeholder =
      MODE === "trajectories" ? "Search trajectories…" :
      MODE === "agentic" ? "Search agentic runs…" :
      MODE === "training" ? "Search training examples…" :
      "Search title, slug, domain…";
  }

  // ---------- views ----------
  function hideAll() {
    hero.hidden = true; article.hidden = true; traj.hidden = true;
    agentic.hidden = true; train.hidden = true;
  }

  function showHero() {
    hideAll();
    hero.hidden = false;
    document.title = "Innovation Prior — reconstructed reasoning traces";
  }

  function showArticle(m) {
    hideAll();
    article.hidden = false;
    $("article-domain").textContent = m.domain;
    $("article-title").textContent = m.title;
    $("article-slug").textContent = m.slug;
    var ax = $("article-arxiv");
    if (m.arxiv) {
      ax.hidden = false;
      ax.href = "https://arxiv.org/abs/" + m.arxiv;
      ax.textContent = "arXiv:" + m.arxiv + " ↗";
    } else {
      ax.hidden = true;
      ax.removeAttribute("href");
      ax.textContent = "";
    }
    document.title = m.title + " — Innovation Prior";
    syncTabs();
  }

  function showTraj(t) {
    hideAll();
    traj.hidden = false;
    $("traj-domain").textContent = t.domain;
    $("traj-title").textContent = t.title;
    $("traj-slug").textContent = t.task;
    document.title = t.title + " — Innovation Prior";
  }

  function syncTabs() {
    article.querySelectorAll(".tab").forEach(function (t) {
      var active = t.getAttribute("data-tab") === current.tab;
      t.classList.toggle("is-active", active);
      t.setAttribute("aria-selected", String(active));
    });
  }

  // ---------- loaders: methods ----------
  function loadContent(m, tab) {
    var key = m.slug + "/" + tab;
    var url = "methods/" + m.slug + "/results/" + tab + ".md";
    if (mdCache[key]) { renderInto(contentEl, mdCache[key]); return; }

    var token = ++fetchToken;
    setStatus(contentEl, "loading", "Loading…");
    fetch(url, { cache: "no-cache" })
      .then(function (res) { if (!res.ok) throw new Error("HTTP " + res.status); return res.text(); })
      .then(function (text) {
        if (token !== fetchToken) return;
        mdCache[key] = text;
        renderInto(contentEl, text);
      })
      .catch(function (err) {
        if (token !== fetchToken) return;
        setStatus(contentEl, "error", "Could not load " + url + " — " + err.message + "." + fetchHint());
      });
  }

  // ---------- loaders: trajectories ----------
  function trajBlocks(t, meta, base) {
    var blocks = [];
    if (meta.initial_context_file) {
      blocks.push({ kind: "context", label: "Initial context", url: base + meta.initial_context_file });
    } else if (meta.initial_context) {
      blocks.push({
        kind: "context", label: "Initial context · " + meta.initial_context,
        url: "methods/" + meta.initial_context + "/results/context.md"
      });
    }
    var nBaselines = (meta.steps || []).filter(function (s) { return !s.finale; }).length;
    var bi = 0;
    (meta.steps || []).forEach(function (s) {
      var who = s.method || s.slug;
      var fin = !!s.finale;
      var stepTag = fin ? ("Finale · " + who + " · stronger published method")
                        : ("Baseline " + (++bi) + "/" + nBaselines + " · " + who);
      blocks.push({
        kind: "reasoning", finale: fin, label: stepTag + " · Reasoning",
        url: s.reasoning ? (base + s.reasoning) : ("methods/" + s.slug + "/results/reasoning.md")
      });
      if (s.answer) blocks.push({ kind: "answer", finale: fin, label: stepTag + " · Answer", url: base + s.answer });
      if (s.feedback) {
        blocks.push({
          kind: "feedback", finale: fin,
          label: fin ? (stepTag + " · the bar to beat") : ("Feedback after baseline " + bi),
          url: base + s.feedback
        });
      }
    });
    return blocks;
  }

  function loadTrajectory(t) {
    var base = "trajectories/" + t.task + "/";
    var token = ++fetchToken;
    setStatus(trajContentEl, "loading", "Loading trajectory…");

    fetch(base + "meta.json", { cache: "no-cache" })
      .then(function (res) { if (!res.ok) throw new Error("meta.json HTTP " + res.status); return res.json(); })
      .then(function (meta) {
        var blocks = trajBlocks(t, meta, base);
        if (!blocks.length) throw new Error("meta.json produced no steps");
        return Promise.all(blocks.map(function (b) {
          return fetch(b.url, { cache: "no-cache" })
            .then(function (r) { return r.ok ? r.text() : "*Missing: `" + b.url + "`*"; })
            .then(function (text) { return { block: b, text: text }; });
        }));
      })
      .then(function (parts) {
        if (token !== fetchToken) return;
        var html = parts.map(function (p) {
          return '<section class="traj-step traj-' + p.block.kind + (p.block.finale ? " traj-is-finale" : "") + '">' +
            '<div class="traj-badge">' + escapeHtml(p.block.label) + "</div>" +
            '<div class="traj-step-body">' + parseMd(p.text) + "</div></section>";
        }).join("");
        trajContentEl.innerHTML = html;
        renderMath(trajContentEl);
        highlightCode(trajContentEl);
        trajContentEl.scrollTop = 0;
      })
      .catch(function (err) {
        if (token !== fetchToken) return;
        setStatus(trajContentEl, "error", "Could not load trajectory " + t.task + " — " + err.message + "." + fetchHint());
      });
  }

  // ---------- shared transcript rendering ----------
  function thinkBlock(text) {
    return '<details class="tx-think" open><summary>Reasoning</summary>' +
      '<div class="tx-think-body">' + parseMd(text) + "</div></details>";
  }

  function callCard(name, args) {
    var argStr;
    if (typeof args === "string") {
      try { argStr = JSON.stringify(JSON.parse(args), null, 2); }
      catch (e) { argStr = args; }
    } else {
      try { argStr = JSON.stringify(args, null, 2); } catch (e) { argStr = String(args); }
    }
    return '<div class="tx-call">' +
      '<div class="tx-call-head"><span class="tx-call-dot"></span>' +
      '<span class="tx-call-name">' + escapeHtml(name || "tool") + "</span>" +
      '<span class="tx-call-tag">tool call</span></div>' +
      '<pre class="tx-call-args"><code class="language-json">' + escapeHtml(argStr) + "</code></pre></div>";
  }

  function toolsBlock(tools) {
    // ShareGPT stores `tools` as a JSON string; agentic transcripts use a real array.
    if (typeof tools === "string") {
      try { tools = JSON.parse(tools); } catch (e) { return ""; }
    }
    if (!Array.isArray(tools) || !tools.length) return "";
    var items = tools.map(function (t) {
      var f = t.function || t;
      var name = f.name || "?";
      var desc = f.description || "";
      var props = (f.parameters && f.parameters.properties) ? Object.keys(f.parameters.properties) : [];
      return "<li><code>" + escapeHtml(name) + "(" + escapeHtml(props.join(", ")) + ")</code>" +
        (desc ? '<span class="tx-tool-desc"> — ' + escapeHtml(desc) + "</span>" : "") + "</li>";
    }).join("");
    return '<details class="tx-block tx-tools"><summary>' + tools.length +
      " tool" + (tools.length > 1 ? "s" : "") + " declared</summary><ul>" + items + "</ul></details>";
  }

  function sysBlock(text) {
    if (!text) return "";
    return '<details class="tx-block tx-sys"><summary>System prompt</summary>' +
      '<div class="tx-block-body">' + parseMd(text) + "</div></details>";
  }

  function bubble(kind, label, inner) {
    return '<div class="tx-msg tx-' + kind + '">' +
      '<div class="tx-role">' + label + "</div>" +
      '<div class="tx-body">' + inner + "</div></div>";
  }

  // ---------- loaders: agentic ----------
  function renderAgenticHtml(rec) {
    var html = "";
    var sys = "";
    (rec.messages || []).forEach(function (m) { if (m.role === "system" && !sys) sys = m.content || ""; });
    html += sysBlock(sys);
    html += toolsBlock(rec.tools);

    var step = 0;
    (rec.messages || []).forEach(function (m) {
      if (m.role === "system") return;
      if (m.role === "user") {
        html += bubble("user", "User", parseMd(m.content || ""));
      } else if (m.role === "assistant") {
        step += 1;
        var inner = "";
        var rc = (m.reasoning_content || "").trim();
        if (rc) inner += thinkBlock(rc);
        var say = (m.content || "").trim();
        if (say) inner += '<div class="tx-say">' + parseMd(say) + "</div>";
        (m.tool_calls || []).forEach(function (tc) {
          var f = tc.function || {};
          inner += callCard(f.name, f.arguments);
        });
        if (!inner) inner = '<div class="tx-say tx-empty">(no output)</div>';
        html += bubble("assistant", "Assistant <span class='tx-step'>step " + step + "</span>", inner);
      } else if (m.role === "tool") {
        html += bubble("tool", "Tool result", parseMd(m.content || ""));
      }
    });
    return html;
  }

  function showAgentic(a) {
    hideAll();
    agentic.hidden = false;
    $("ag-domain").textContent = a.domain;
    $("ag-title").textContent = a.title;
    $("ag-task").textContent = a.task;
    $("ag-meta").innerHTML =
      (a.year ? '<span class="tx-pill">' + a.year + "</span>" : "") +
      '<span class="tx-pill">' + a.n_steps + " steps</span>" +
      '<span class="tx-pill">' + a.n_actions + " tool calls</span>" +
      '<span class="tx-pill">' + a.n_tools + " tools</span>";
    document.title = a.title + " — Innovation Prior (agentic)";
  }

  function loadAgentic(a) {
    if (agCache[a.task]) { paintHtml(agContentEl, renderAgenticHtml(agCache[a.task])); return; }
    var url = "trajectories/" + a.task + "/agentic_messages.json";
    var token = ++fetchToken;
    setStatus(agContentEl, "loading", "Loading agentic transcript…");
    fetch(url, { cache: "no-cache" })
      .then(function (res) { if (!res.ok) throw new Error("HTTP " + res.status); return res.json(); })
      .then(function (rec) {
        if (token !== fetchToken) return;
        agCache[a.task] = rec;
        paintHtml(agContentEl, renderAgenticHtml(rec));
      })
      .catch(function (err) {
        if (token !== fetchToken) return;
        setStatus(agContentEl, "error", "Could not load " + url + " — " + err.message + "." + fetchHint());
      });
  }

  // ---------- loaders: training data ----------
  function ensureTrainIndex() {
    if (trainIndex) return Promise.resolve(trainIndex);
    if (trainIndexPromise) return trainIndexPromise;
    trainIndexPromise = fetch("sft/viewer/index.json", { cache: "no-cache" })
      .then(function (res) { if (!res.ok) throw new Error("HTTP " + res.status); return res.json(); })
      .then(function (j) {
        trainIndex = j;
        (j.examples || []).forEach(function (e) {
          trainById[e.id] = e;
          var g = trainGroupOf(e);
          if (!(g in collapsed)) collapsed[g] = true; // training groups start collapsed
        });
        return j;
      });
    return trainIndexPromise;
  }

  // Fetch a gzipped shard and gunzip in-browser (pako). Robust to servers that
  // already decode .gz (Content-Encoding: gzip): we sniff the gzip magic bytes.
  function loadShard(name) {
    if (shardCache[name]) return Promise.resolve(shardCache[name]);
    var url = "sft/viewer/" + name;
    return fetch(url, { cache: "no-cache" })
      .then(function (res) { if (!res.ok) throw new Error("HTTP " + res.status); return res.arrayBuffer(); })
      .then(function (buf) {
        var u8 = new Uint8Array(buf);
        var text;
        if (u8.length > 2 && u8[0] === 0x1f && u8[1] === 0x8b) {
          if (typeof pako === "undefined") throw new Error("pako (gunzip) library not loaded");
          text = pako.ungzip(u8, { to: "string" });
        } else {
          text = new TextDecoder("utf-8").decode(u8); // server already decompressed
        }
        var arr = JSON.parse(text);
        shardCache[name] = arr;
        return arr;
      });
  }

  // Parse an assistant turn value: pull out <think>…</think>, <tool_call>…</tool_call>
  // (JSON), leaving the spoken text. Mirrors how the SFT data encodes a turn.
  function parseAssistantValue(val) {
    var think = null, rest = String(val), calls = [];
    var tm = rest.match(/<think>([\s\S]*?)<\/think>/);
    if (tm) { think = tm[1].trim(); rest = rest.replace(tm[0], ""); }
    rest = rest.replace(/<tool_call>([\s\S]*?)<\/tool_call>/g, function (_, inner) {
      var j = inner.trim(), name = "tool", args = j;
      try { var o = JSON.parse(j); name = o.name || "tool"; args = (o.arguments !== undefined) ? o.arguments : o; }
      catch (e) { /* keep raw */ }
      calls.push({ name: name, args: args });
      return "";
    });
    return { think: think, say: rest.trim(), calls: calls };
  }

  function lossTag(loss) {
    if (loss === false) return '<span class="tx-tag tx-masked">masked · no loss</span>';
    if (loss === true) return '<span class="tx-tag tx-trained">trained</span>';
    return "";
  }

  function trainMetaStrip(ex, entry) {
    var pills = [];
    pills.push('<span class="tx-pill tx-pill-ds">' + escapeHtml(entry.ds) + "</span>");
    pills.push('<span class="tx-pill">' + escapeHtml(entry.kind) + "</span>");
    if (entry.year) pills.push('<span class="tx-pill">' + entry.year + "</span>");
    pills.push('<span class="tx-pill">' + entry.turns + " user · " + entry.actions + " calls</span>");
    if (entry.thinking === false || ex.enable_thinking === false)
      pills.push('<span class="tx-pill tx-pill-warn">enable_thinking: false</span>');
    else
      pills.push('<span class="tx-pill">thinking</span>');
    if (entry.masked) pills.push('<span class="tx-pill tx-pill-warn">per-turn loss folding</span>');
    return '<div class="tx-meta">' + pills.join("") + "</div>";
  }

  function renderTrainExampleHtml(ex, entry) {
    var html = trainMetaStrip(ex, entry);
    if (ex.system) html += sysBlock(ex.system);
    if (ex.tools && ex.tools.length) html += toolsBlock(ex.tools);

    (ex.conversations || []).forEach(function (c) {
      var from = c.from, val = c.value || "";
      if (from === "human") {
        html += bubble("user", "User", parseMd(val));
      } else if (from === "observation") {
        html += bubble("tool", "Tool result", parseMd(val));
      } else { // gpt | function_call (assistant)
        var p = parseAssistantValue(val);
        var inner = "";
        if (p.think) inner += thinkBlock(p.think);
        if (p.say) inner += '<div class="tx-say">' + parseMd(p.say) + "</div>";
        p.calls.forEach(function (cc) { inner += callCard(cc.name, cc.args); });
        if (!inner) inner = '<div class="tx-say tx-empty">(empty think — placed in prompt, not trained)</div>';
        var tag = ("loss" in c) ? lossTag(c.loss) : "";
        html += bubble("assistant", "Assistant " + tag, inner);
      }
    });
    return html;
  }

  function trainHomeHtml() {
    var d = trainIndex.datasets || {};
    function table(name, info) {
      if (!info) return "";
      var rows = Object.keys(info.by_kind || {}).map(function (k) {
        return "<tr><td>" + escapeHtml(k) + "</td><td>" + info.by_kind[k] + "</td></tr>";
      }).join("");
      return '<div class="td-card"><h3>' + escapeHtml(name) + ' <span class="td-count">' + info.count +
        " examples · " + info.shards + " shards</span></h3>" +
        '<table class="td-table"><thead><tr><th>kind</th><th>count</th></tr></thead><tbody>' +
        rows + "</tbody></table></div>";
    }
    return '<div class="td-home">' +
      "<p>The reasoning traces are assembled into a <strong>supervised fine-tuning corpus</strong> " +
      "(LLaMA-Factory ShareGPT format). Every one of the <strong>" + trainIndex.total +
      "</strong> training examples is browsable below — pick one from the sidebar to inspect its " +
      "conversation, tool calls, and the per-turn <em>loss</em> / <em>enable_thinking</em> metadata that " +
      "lets reasoning, non-reasoning, and folded-history data all train in one run.</p>" +
      '<div class="td-cards">' +
      table("innovation_sft", d.innovation_sft) +
      table("maintain_sft", d.maintain_sft) +
      "</div>" +
      '<p class="td-note">Bodies lazy-load from gzipped shards (<code>sft/viewer/*.json.gz</code>) and are ' +
      "gunzipped in your browser. Full build + training docs: <code>sft/README.md</code>.</p>" +
      "</div>";
  }

  function showTrainHome() {
    hideAll();
    train.hidden = false;
    $("train-domain").textContent = "Training data";
    $("train-title").textContent = "SFT corpus — all " + (trainIndex ? trainIndex.total : "") + " examples";
    $("train-badges").innerHTML = "";
    $("train-id").textContent = "";
    document.title = "Training data — Innovation Prior";
    paintHtml(trainContentEl, trainHomeHtml());
  }

  function showTrainExample(id) {
    var entry = trainById[id];
    hideAll();
    train.hidden = false;
    $("train-domain").textContent = entry.ds;
    $("train-title").textContent = entry.title;
    $("train-id").textContent = "#" + entry.id;
    $("train-badges").innerHTML = '<span class="traj-tag">Training example</span>';
    document.title = "Training #" + entry.id + " — Innovation Prior";

    var token = ++fetchToken;
    setStatus(trainContentEl, "loading", "Loading example…");
    loadShard(entry.shard)
      .then(function (arr) {
        if (token !== fetchToken) return;
        var ex = arr[entry.i];
        if (!ex) throw new Error("example missing in shard " + entry.shard);
        paintHtml(trainContentEl, renderTrainExampleHtml(ex, entry));
      })
      .catch(function (err) {
        if (token !== fetchToken) return;
        setStatus(trainContentEl, "error", "Could not load example #" + entry.id + " — " + err.message + "." + fetchHint());
      });
  }

  // ---------- routing ----------
  function parseHash() {
    var h = location.hash.replace(/^#\/?/, "");
    if (h === "" || h === "m") return { mode: "methods", slug: null, tab: DEFAULT_TAB };
    var parts = h.split("/");
    if (parts[0] === "t") return { mode: "trajectories", task: parts[1] || null };
    if (parts[0] === "a") return { mode: "agentic", agtask: parts[1] || null };
    if (parts[0] === "d") return { mode: "training", id: (parts[1] != null && parts[1] !== "") ? parts[1] : null };
    var slug = parts[0] || null;
    var tab = parts[1];
    if (TABS.indexOf(tab) === -1) tab = DEFAULT_TAB;
    return { mode: "methods", slug: slug, tab: tab };
  }

  function route() {
    var r = parseHash();
    MODE = r.mode;
    updateModeButtons();

    if (MODE === "training") { routeTraining(r); return; }
    if (builtMode !== MODE) buildSidebar(searchInput.value);

    if (MODE === "trajectories") {
      if (!r.task || !trajByTask[r.task]) { current.task = null; showHero(); markActive(); return; }
      var t = trajByTask[r.task];
      var tChanged = current.task !== r.task;
      current.task = r.task;
      showTraj(t); markActive(); loadTrajectory(t);
      if (tChanged) closeMobileNav();
      return;
    }

    if (MODE === "agentic") {
      if (!r.agtask || !agByTask[r.agtask]) { current.agtask = null; showHero(); markActive(); return; }
      var a = agByTask[r.agtask];
      var aChanged = current.agtask !== r.agtask;
      current.agtask = r.agtask;
      showAgentic(a); markActive(); loadAgentic(a);
      if (aChanged) closeMobileNav();
      return;
    }

    // methods mode
    if (!r.slug || !bySlug[r.slug]) { current.slug = null; showHero(); markActive(); return; }
    var m = bySlug[r.slug];
    var slugChanged = current.slug !== r.slug;
    current.slug = r.slug;
    current.tab = r.tab;
    showArticle(m); markActive(); loadContent(m, r.tab);
    if (slugChanged) closeMobileNav();
  }

  function routeTraining(r) {
    ensureTrainIndex()
      .then(function () {
        if (builtMode !== "training") buildSidebar(searchInput.value);
        if (r.id == null || !trainById[r.id]) { current.id = null; showTrainHome(); markActive(); return; }
        var changed = current.id !== r.id;
        current.id = r.id;
        showTrainExample(r.id); markActive();
        if (changed) closeMobileNav();
      })
      .catch(function (err) {
        hideAll(); train.hidden = false;
        $("train-domain").textContent = "Training data";
        $("train-title").textContent = "SFT corpus";
        setStatus(trainContentEl, "error", "Failed to load training index — " + err.message + "." + fetchHint());
      });
  }

  // ---------- mobile nav ----------
  function openMobileNav() {
    sidebar.classList.add("is-open");
    scrim.hidden = false;
    menuToggle.setAttribute("aria-expanded", "true");
  }
  function closeMobileNav() {
    sidebar.classList.remove("is-open");
    scrim.hidden = true;
    menuToggle.setAttribute("aria-expanded", "false");
  }
  function toggleMobileNav() {
    if (sidebar.classList.contains("is-open")) closeMobileNav();
    else openMobileNav();
  }

  // ---------- theme ----------
  function syncHljsTheme() {
    var dark = window.matchMedia &&
      window.matchMedia("(prefers-color-scheme: dark)").matches;
    var light = $("hljs-light");
    var darkSheet = $("hljs-dark");
    if (light) light.disabled = !!dark;
    if (darkSheet) darkSheet.disabled = !dark;
  }

  // ---------- init ----------
  function wireEvents() {
    article.querySelectorAll(".tab").forEach(function (t) {
      t.addEventListener("click", function () {
        if (!current.slug) return;
        location.hash = "#" + current.slug + "/" + t.getAttribute("data-tab");
      });
    });

    modeBtns.methods.addEventListener("click", function () { location.hash = "#m"; });
    modeBtns.trajectories.addEventListener("click", function () { location.hash = "#t"; });
    if (modeBtns.agentic) modeBtns.agentic.addEventListener("click", function () { location.hash = "#a"; });
    if (modeBtns.training) modeBtns.training.addEventListener("click", function () { location.hash = "#d"; });

    searchInput.addEventListener("input", function () { buildSidebar(searchInput.value); });

    document.addEventListener("keydown", function (e) {
      if (e.key === "/" && document.activeElement !== searchInput) {
        e.preventDefault();
        searchInput.focus();
        searchInput.select();
      } else if (e.key === "Escape" && document.activeElement === searchInput) {
        searchInput.value = "";
        buildSidebar("");
        searchInput.blur();
      }
    });

    menuToggle.addEventListener("click", toggleMobileNav);
    scrim.addEventListener("click", closeMobileNav);
    window.addEventListener("hashchange", route);

    if (window.matchMedia) {
      var mq = window.matchMedia("(prefers-color-scheme: dark)");
      if (mq.addEventListener) mq.addEventListener("change", syncHljsTheme);
      else if (mq.addListener) mq.addListener(syncHljsTheme);
    }
  }

  function init() {
    configureMarked();
    syncHljsTheme();
    wireEvents();

    var pMethods = fetch("methods.json", { cache: "no-cache" })
      .then(function (res) { if (!res.ok) throw new Error("HTTP " + res.status); return res.json(); })
      .then(function (data) {
        methods = Array.isArray(data) ? data : [];
        methods.forEach(function (m) { bySlug[m.slug] = m; collapsed[m.domain] = false; });
        var countEl = $("hero-count");
        if (countEl) countEl.textContent = String(methods.length);
      });

    var pTraj = fetch("trajectories.json", { cache: "no-cache" })
      .then(function (res) { return res.ok ? res.json() : []; })
      .then(function (data) {
        trajectories = Array.isArray(data) ? data : [];
        trajectories.forEach(function (t) {
          trajByTask[t.task] = t;
          if (!(t.domain in collapsed)) collapsed[t.domain] = false;
        });
        var tc = $("hero-traj-count");
        if (tc) tc.textContent = String(trajectories.length);
      })
      .catch(function () { trajectories = []; });

    var pAg = fetch("agentic.json", { cache: "no-cache" })
      .then(function (res) { return res.ok ? res.json() : []; })
      .then(function (data) {
        agentics = Array.isArray(data) ? data : [];
        agentics.forEach(function (a) {
          agByTask[a.task] = a;
          if (!(a.domain in collapsed)) collapsed[a.domain] = false;
        });
        var ac = $("hero-ag-count");
        if (ac) ac.textContent = String(agentics.length);
      })
      .catch(function () { agentics = []; });

    Promise.all([pMethods, pTraj, pAg])
      .then(function () { buildSidebar(""); route(); })
      .catch(function (err) {
        nav.innerHTML = '<p class="nav-empty">Failed to load methods.json (' +
          escapeHtml(err.message) + ").</p>";
        setStatus(contentEl, "error", "Failed to load methods.json: " + err.message + "." + fetchHint());
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
