/* Innovation Prior — data-driven single-page app.
 * Two browse modes:
 *  - Methods:      atomic reasoning traces. methods.json -> #<slug>/<tab>,
 *                  fetches methods/<slug>/results/<tab>.md (Context/Reasoning/Answer).
 *  - Trajectories: iterative research lines over an MLS-Bench task. trajectories.json
 *                  -> #t/<task>, reads trajectories/<task>/meta.json then walks its
 *                  ordered files (initial context -> baselines -> feedback -> finale).
 * Rendering: markdown (marked) -> math (KaTeX) -> code (highlight.js).
 */
(function () {
  "use strict";

  var TABS = ["context", "reasoning", "answer"];
  var DEFAULT_TAB = "reasoning";

  var methods = [];           // [{slug,title,domain,arxiv}]
  var bySlug = {};            // slug -> method
  var trajectories = [];      // [{task,title,domain,finale}]
  var trajByTask = {};        // task -> trajectory
  var collapsed = {};         // domain -> bool
  var MODE = "methods";       // "methods" | "trajectories"
  var builtMode = null;       // which mode the sidebar DOM currently reflects
  var current = { slug: null, tab: DEFAULT_TAB, task: null };
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
  var sidebar = $("sidebar");
  var scrim = $("scrim");
  var menuToggle = $("menu-toggle");
  var modeMethodsBtn = $("mode-methods");
  var modeTrajBtn = $("mode-traj");

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

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, function (c) {
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

  // ---------- sidebar ----------
  function groupByDomain(list) {
    var groups = [];
    var index = {};
    list.forEach(function (m) {
      if (!(m.domain in index)) {
        index[m.domain] = groups.length;
        groups.push({ domain: m.domain, items: [] });
      }
      groups[index[m.domain]].items.push(m);
    });
    return groups;
  }

  function buildSidebar(filter) {
    var q = (filter || "").trim().toLowerCase();
    var isTraj = MODE === "trajectories";
    var source = isTraj ? trajectories : methods;
    var keyOf = function (m) { return isTraj ? m.task : m.slug; };

    var matches = source.filter(function (m) {
      if (!q) return true;
      var hay = (m.title + " " + keyOf(m) + " " + m.domain + " " + (m.finale || "")).toLowerCase();
      return hay.indexOf(q) !== -1;
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

    var activeKey = isTraj ? current.task : current.slug;

    groupByDomain(matches).forEach(function (group) {
      var section = document.createElement("div");
      section.className = "nav-group";

      var isCollapsed = !q && collapsed[group.domain]; // searching forces expand
      var btn = document.createElement("button");
      btn.className = "nav-group-head";
      btn.type = "button";
      btn.setAttribute("aria-expanded", String(!isCollapsed));
      btn.innerHTML =
        '<span class="caret" aria-hidden="true">▸</span>' +
        '<span class="nav-group-name">' + escapeHtml(group.domain) + "</span>" +
        '<span class="nav-group-count">' + group.items.length + "</span>";
      btn.addEventListener("click", function () {
        collapsed[group.domain] = !collapsed[group.domain];
        buildSidebar(searchInput.value);
      });
      section.appendChild(btn);

      var ul = document.createElement("ul");
      ul.className = "nav-list";
      if (isCollapsed) ul.hidden = true;

      group.items.forEach(function (m) {
        var k = keyOf(m);
        var li = document.createElement("li");
        var a = document.createElement("a");
        a.className = "nav-link" + (k === activeKey ? " is-active" : "");
        a.href = isTraj ? ("#t/" + k) : ("#" + k + "/" + current.tab);
        a.setAttribute(isTraj ? "data-task" : "data-slug", k);
        var sub = (isTraj && m.finale)
          ? '<span class="nav-link-sub">→ ' + escapeHtml(m.finale) + "</span>"
          : "";
        a.innerHTML = '<span class="nav-link-title">' + escapeHtml(m.title) + "</span>" + sub;
        if (k === activeKey) a.setAttribute("aria-current", "page");
        li.appendChild(a);

        if (!isTraj && m.arxiv) {
          var ext = document.createElement("a");
          ext.className = "nav-arxiv";
          ext.href = "https://arxiv.org/abs/" + m.arxiv;
          ext.target = "_blank";
          ext.rel = "noopener";
          ext.title = "arXiv:" + m.arxiv;
          ext.textContent = m.arxiv;
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
    var isTraj = MODE === "trajectories";
    var cur = isTraj ? current.task : current.slug;
    nav.querySelectorAll(".nav-link").forEach(function (a) {
      var k = a.getAttribute(isTraj ? "data-task" : "data-slug");
      var active = k != null && k === cur;
      a.classList.toggle("is-active", active);
      if (active) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
      if (!isTraj && a.getAttribute("data-slug")) {
        a.href = "#" + a.getAttribute("data-slug") + "/" + current.tab;
      }
    });
  }

  function updateModeButtons() {
    modeMethodsBtn.classList.toggle("is-active", MODE === "methods");
    modeTrajBtn.classList.toggle("is-active", MODE === "trajectories");
    searchInput.placeholder = (MODE === "trajectories")
      ? "Search trajectories…" : "Search title, slug, domain…";
  }

  // ---------- views ----------
  function hideAll() { hero.hidden = true; article.hidden = true; traj.hidden = true; }

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

  // ---------- loaders ----------
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

  // Build the ordered list of render blocks for a trajectory from its meta.json.
  // The reasoning/answer/context blocks REUSE the full methods/<slug>/ traces (no
  // duplication); only the reflection + numbers-only feedback are trajectory-local.
  function trajBlocks(t, meta, base) {
    var blocks = [];
    if (meta.initial_context_file) {
      // MLS-Bench-based trajectory: the initial context is authored locally (the task's scaffold).
      blocks.push({
        kind: "context",
        label: "Initial context",
        url: base + meta.initial_context_file
      });
    } else if (meta.initial_context) {
      // Otherwise reuse a baseline's methods/ context verbatim.
      blocks.push({
        kind: "context",
        label: "Initial context · " + meta.initial_context,
        url: "methods/" + meta.initial_context + "/results/context.md"
      });
    }
    (meta.steps || []).forEach(function (s) {
      var who = s.method || s.slug;
      // Multi-round reasoning (with the reflection on the previous result embedded) lives in the
      // trajectory when authored; step 1 falls back to the single-round methods/<slug> trace.
      blocks.push({
        kind: "reasoning",
        label: "Step " + s.n + " · " + who + " · Reasoning",
        url: s.reasoning ? (base + s.reasoning) : ("methods/" + s.slug + "/results/reasoning.md")
      });
      if (s.feedback) {
        blocks.push({
          kind: "feedback",
          label: s.finale ? ("Step " + s.n + " · " + who + " · the bar to beat") : ("Feedback after step " + s.n),
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
          return '<section class="traj-step traj-' + p.block.kind + '">' +
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

  // ---------- routing ----------
  function parseHash() {
    var h = location.hash.replace(/^#\/?/, "");
    if (h === "" || h === "m") return { mode: "methods", slug: null, tab: DEFAULT_TAB, task: null };
    var parts = h.split("/");
    if (parts[0] === "t") return { mode: "trajectories", task: parts[1] || null };
    var slug = parts[0] || null;
    var tab = parts[1];
    if (TABS.indexOf(tab) === -1) tab = DEFAULT_TAB;
    return { mode: "methods", slug: slug, tab: tab };
  }

  function route() {
    var r = parseHash();
    MODE = r.mode;
    updateModeButtons();
    if (builtMode !== MODE) buildSidebar(searchInput.value);

    if (MODE === "trajectories") {
      if (!r.task) { current.task = null; showHero(); markActive(); return; }
      var t = trajByTask[r.task];
      if (!t) { current.task = null; showHero(); markActive(); return; }
      var tChanged = current.task !== r.task;
      current.task = r.task;
      showTraj(t);
      markActive();
      loadTrajectory(t);
      if (tChanged) closeMobileNav();
      return;
    }

    // methods mode
    if (!r.slug) { current.slug = null; showHero(); markActive(); return; }
    var m = bySlug[r.slug];
    if (!m) { current.slug = null; showHero(); markActive(); return; }
    var slugChanged = current.slug !== r.slug;
    current.slug = r.slug;
    current.tab = r.tab;
    showArticle(m);
    markActive();
    loadContent(m, r.tab);
    if (slugChanged) closeMobileNav();
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

  // ---------- theme (sync hljs stylesheet to scheme) ----------
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

    modeMethodsBtn.addEventListener("click", function () { location.hash = "#m"; });
    modeTrajBtn.addEventListener("click", function () { location.hash = "#t"; });

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

    // trajectories.json is optional — absence just leaves that mode's list empty.
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

    Promise.all([pMethods, pTraj])
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
