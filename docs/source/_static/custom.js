(function () {
  var root = document.documentElement, KEY = "gw-theme";
  var mq = window.matchMedia("(prefers-color-scheme: dark)");
  var MODES = ["auto", "light", "dark"];
  var btn = null;

  var ICONS = {
    light: '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>',
    dark: '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5z"/></svg>'
  };
  var LABEL = { auto: "Theme: follow system", light: "Theme: light", dark: "Theme: dark" };

  function stored() {
    try { var v = localStorage.getItem(KEY); return MODES.indexOf(v) >= 0 ? v : "auto"; }
    catch (e) { return "auto"; }
  }
  function effective(mode) {
    return mode === "auto" ? (mq.matches ? "dark" : "light") : mode;
  }
  function paint(mode) {
    if (!btn) return;
    btn.innerHTML = ICONS[effective(mode)];   // sun or moon, even in auto
    btn.setAttribute("data-mode", mode);      // CSS adds the "A" badge for auto
    btn.title = LABEL[mode] + " (click to change)";
    btn.setAttribute("aria-label", LABEL[mode]);
  }
  function apply(mode) {
    root.setAttribute("data-theme", effective(mode));
    root.setAttribute("data-theme-mode", mode);
    paint(mode);
    window.dispatchEvent(new CustomEvent("themechange", { detail: effective(mode) }));
  }

  var mode = stored();
  apply(mode);
  mq.addEventListener("change", function () { if (mode === "auto") apply("auto"); });

  document.addEventListener("DOMContentLoaded", function () {
    var holder = document.querySelector(".wy-side-nav-search");
    if (holder) {
      btn = document.createElement("button");
      btn.className = "theme-toggle";
      btn.type = "button";
      paint(mode);
      btn.onclick = function () {
        mode = MODES[(MODES.indexOf(mode) + 1) % MODES.length];
        try { localStorage.setItem(KEY, mode); } catch (e) {}
        apply(mode);
      };
      holder.appendChild(btn);
    }

    // pause videos when offscreen
    var vids = document.querySelectorAll(".gw-card video");
    if ("IntersectionObserver" in window && vids.length) {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting) e.target.play().catch(function () {});
          else e.target.pause();
        });
      }, { threshold: 0.25 });
      vids.forEach(function (v) { io.observe(v); });
    }
  });
})();