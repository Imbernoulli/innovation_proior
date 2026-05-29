/* Innovation Prior — data-driven single-page app.
 * - Loads methods.json
 * - Sidebar grouped by domain (collapsible) + search
 * - Three tabs (Context / Reasoning / Answer), default Reasoning
 * - Fetches `methods/${slug}/results/${tab}.md`, renders markdown -> KaTeX -> highlight
 * - Hash routing: #<slug>/<tab>, back/forward friendly
 */
(function () {
  "use strict";

  var TABS = ["context", "reasoning", "answer"];
  var DEFAULT_TAB = "reasoning";

  var methods = [];           // [{slug,title,domain,arxiv}]
  var bySlug = {};            // slug -> method
  var collapsed = {};         // domain -> bool
  var current = { slug: null, tab: DEFAULT_TAB };
  var fetchToken = 0;         // guards against out-of-order responses
  var mdCache = {};           // "slug/tab" -> html

  // ---- DOM refs ----
  var $ = function (id) { return document.getElementById(id); };
  var nav = $("nav");
  var searchInput = $("search");
  var hero = $("hero");
  var article = $("article");
  var contentEl = $("content");
  var sidebar = $("sidebar");
  var scrim = $("scrim");
  var menuToggle = $("menu-toggle");

  // ---------- markdown / math / code rendering ----------
  function configureMarked() {
    if (typeof marked === "undefined") return;
    marked.setOptions({ gfm: true, breaks: false, headerIds: true, mangle: false });
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

  function renderMarkdown(text) {
    var html = (typeof marked !== "undefined")
      ? (marked.parse ? marked.parse(text) : marked(text))
      : ("<pre>" + escapeHtml(text) + "</pre>");
    contentEl.innerHTML = html;
    // Order matters: markdown -> math -> code highlight.
    renderMath(contentEl);
    highlightCode(contentEl);
    contentEl.scrollTop = 0;
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
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
    var matches = methods.filter(function (m) {
      if (!q) return true;
      return (m.title + " " + m.slug + " " + m.domain).toLowerCase().indexOf(q) !== -1;
    });

    nav.innerHTML = "";
    if (matches.length === 0) {
      var empty = document.createElement("p");
      empty.className = "nav-empty";
      empty.textContent = "No methods match “" + filter + "”.";
      nav.appendChild(empty);
      return;
    }

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
        var li = document.createElement("li");

        var a = document.createElement("a");
        a.className = "nav-link" + (m.slug === current.slug ? " is-active" : "");
        a.href = "#" + m.slug + "/" + current.tab;
        a.setAttribute("data-slug", m.slug);
        a.innerHTML =
          '<span class="nav-link-title">' + escapeHtml(m.title) + "</span>";
        if (m.slug === current.slug) a.setAttribute("aria-current", "page");

        var ext = document.createElement("a");
        ext.className = "nav-arxiv";
        ext.href = "https://arxiv.org/abs/" + m.arxiv;
        ext.target = "_blank";
        ext.rel = "noopener";
        ext.title = "arXiv:" + m.arxiv;
        ext.textContent = m.arxiv;
        ext.addEventListener("click", function (e) { e.stopPropagation(); });

        li.appendChild(a);
        li.appendChild(ext);
        ul.appendChild(li);
      });
      section.appendChild(ul);
      nav.appendChild(section);
    });
  }

  function markActiveLink() {
    nav.querySelectorAll(".nav-link").forEach(function (a) {
      var active = a.getAttribute("data-slug") === current.slug;
      a.classList.toggle("is-active", active);
      if (active) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
      // keep tab in nav hrefs current
      if (current.slug) a.href = "#" + a.getAttribute("data-slug") + "/" + current.tab;
    });
  }

  // ---------- views ----------
  function showHero() {
    article.hidden = true;
    hero.hidden = false;
    document.title = "Innovation Prior — reconstructed reasoning traces";
  }

  function showArticle(m) {
    hero.hidden = true;
    article.hidden = false;
    $("article-domain").textContent = m.domain;
    $("article-title").textContent = m.title;
    $("article-slug").textContent = m.slug;
    var ax = $("article-arxiv");
    ax.href = "https://arxiv.org/abs/" + m.arxiv;
    ax.textContent = "arXiv:" + m.arxiv + " ↗";
    document.title = m.title + " — Innovation Prior";
    syncTabs();
  }

  function syncTabs() {
    article.querySelectorAll(".tab").forEach(function (t) {
      var active = t.getAttribute("data-tab") === current.tab;
      t.classList.toggle("is-active", active);
      t.setAttribute("aria-selected", String(active));
    });
  }

  function setStatus(kind, msg) {
    contentEl.innerHTML =
      '<div class="status status-' + kind + '">' + escapeHtml(msg) + "</div>";
  }

  function loadContent(m, tab) {
    var key = m.slug + "/" + tab;
    var url = "methods/" + m.slug + "/results/" + tab + ".md";

    if (mdCache[key]) { renderMarkdown(mdCache[key]); return; }

    var token = ++fetchToken;
    setStatus("loading", "Loading…");

    fetch(url, { cache: "no-cache" })
      .then(function (res) {
        if (!res.ok) throw new Error("HTTP " + res.status);
        return res.text();
      })
      .then(function (text) {
        if (token !== fetchToken) return; // superseded
        mdCache[key] = text;
        renderMarkdown(text);
      })
      .catch(function (err) {
        if (token !== fetchToken) return;
        var hint = (location.protocol === "file:")
          ? " This page must be served over HTTP (e.g. a local server or GitHub Pages); the file:// protocol blocks fetch()."
          : "";
        setStatus(
          "error",
          "Could not load " + url + " — " + err.message + "." + hint
        );
      });
  }

  // ---------- routing ----------
  function parseHash() {
    var h = location.hash.replace(/^#\/?/, "");
    if (!h) return { slug: null, tab: DEFAULT_TAB };
    var parts = h.split("/");
    var slug = parts[0] || null;
    var tab = parts[1];
    if (TABS.indexOf(tab) === -1) tab = DEFAULT_TAB;
    return { slug: slug, tab: tab };
  }

  function route() {
    var r = parseHash();

    if (!r.slug) {
      current.slug = null;
      current.tab = r.tab;
      showHero();
      markActiveLink();
      return;
    }

    var m = bySlug[r.slug];
    if (!m) {
      current.slug = null;
      showHero();
      setStatus("error", "Unknown method “" + r.slug + "”.");
      // still show hero, but also surface message? Keep hero clean: just go home.
      showHero();
      markActiveLink();
      return;
    }

    var slugChanged = current.slug !== r.slug;
    current.slug = r.slug;
    current.tab = r.tab;
    showArticle(m);
    markActiveLink();
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
    // tab clicks -> update hash (keeps history)
    article.querySelectorAll(".tab").forEach(function (t) {
      t.addEventListener("click", function () {
        if (!current.slug) return;
        location.hash = "#" + current.slug + "/" + t.getAttribute("data-tab");
      });
    });

    searchInput.addEventListener("input", function () {
      buildSidebar(searchInput.value);
    });

    // "/" focuses search
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

    fetch("methods.json", { cache: "no-cache" })
      .then(function (res) {
        if (!res.ok) throw new Error("HTTP " + res.status);
        return res.json();
      })
      .then(function (data) {
        methods = Array.isArray(data) ? data : [];
        methods.forEach(function (m) {
          bySlug[m.slug] = m;
          collapsed[m.domain] = false; // expanded by default
        });
        var countEl = $("hero-count");
        if (countEl) countEl.textContent = String(methods.length);
        buildSidebar("");
        route();
      })
      .catch(function (err) {
        nav.innerHTML =
          '<p class="nav-empty">Failed to load methods.json (' +
          escapeHtml(err.message) + ").</p>";
        var hint = (location.protocol === "file:")
          ? " Serve this site over HTTP (e.g. `python3 -m http.server`) — file:// blocks fetch()."
          : "";
        setStatus("error", "Failed to load methods.json: " + err.message + "." + hint);
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
