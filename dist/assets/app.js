/* MineScars UI — talks to the FastAPI backend that serves it:
 *   GET  /api/health                 runtime info
 *   POST /api/process (multipart)    segmentation + Replicate job submission
 *   GET  /api/status?job_url=...     poll the FLUX.1 Fill job
 * No build step: this file is served as-is from dist/. Any Tailwind class used
 * here must appear as a literal string so frontend/ can pick it up when the CSS
 * is rebuilt.
 */
(() => {
  "use strict";

  const MAX_BYTES = 8 * 1024 * 1024; // matches MAX_UPLOAD_MB default in app.py
  const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];
  const POLL_MS = 4000;
  const POLL_MAX_MS = 10 * 60 * 1000;
  const POLL_MAX_ERRORS = 3;

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const state = {
    file: null,
    previewUrl: null,
    busy: false,
    result: null,
    health: null,
    outputUrl: null,
    jobUrl: null,
    poll: null,
    tick: null,
    startedAt: 0,
    p2Mode: "overlay",
    format: "square",
  };

  /* ───────────── helpers ───────────── */

  let toastTimer = null;
  function toast(msg) {
    const el = $("#toast");
    $("#toast-text").textContent = msg;
    el.classList.remove("translate-y-24", "opacity-0");
    el.classList.add("translate-y-0", "opacity-100");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      el.classList.add("translate-y-24", "opacity-0");
      el.classList.remove("translate-y-0", "opacity-100");
    }, 3500);
  }

  function showError(msg) {
    $("#error-text").textContent = msg;
    $("#error-banner").hidden = false;
  }
  function clearError() {
    $("#error-banner").hidden = true;
  }
  $("#error-dismiss").addEventListener("click", clearError);

  function errorDetail(data, res) {
    const d = data && data.detail;
    if (typeof d === "string" && d) return d;
    if (Array.isArray(d) && d.length) return d.map((x) => x.msg || String(x)).join("; ");
    if (res.status === 413) return "That image is too large (8 MB maximum).";
    if (res.status === 429) return "Too many requests. Please wait a few minutes and try again.";
    return `Request failed (${res.status}).`;
  }

  function fmtBytes(n) {
    return n >= 1024 * 1024 ? `${(n / 1024 / 1024).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`;
  }
  const fmtSize = (s) => (s && s.width ? `${s.width} × ${s.height} px` : "—");

  /* ───────────── routing ───────────── */

  const TITLES = { home: "MineScars — See the unseen", lab: "Test Lab — MineScars", results: "Results — MineScars" };

  function currentRoute() {
    const name = location.hash.replace(/^#\/?/, "").split("?")[0] || "home";
    return ["home", "lab", "results"].includes(name) ? name : "home";
  }

  function route() {
    let name = currentRoute();
    if (name === "results" && !state.result) {
      location.replace("#/lab");
      return;
    }
    $$("[data-view]").forEach((v) => { v.hidden = v.dataset.view !== name; });
    $$("[data-nav]").forEach((a) => {
      if (a.dataset.nav === name) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
    });
    $('[data-nav="results"]').hidden = !state.result;
    document.title = TITLES[name];
    clearError();
    window.scrollTo(0, 0);
  }
  window.addEventListener("hashchange", route);

  $$("[data-scroll]").forEach((b) =>
    b.addEventListener("click", () => {
      const t = document.getElementById(b.dataset.scroll);
      if (t) t.scrollIntoView({ behavior: "smooth" });
    })
  );

  /* ───────────── backend health ───────────── */

  function renderRuntime() {
    const h = state.health;
    const dot = $("#runtime-dot");
    const text = $("#runtime-text");
    if (!h) {
      dot.className = "w-1.5 h-1.5 rounded-full bg-error";
      text.textContent = "Backend offline";
      return;
    }
    dot.className = "w-1.5 h-1.5 rounded-full bg-secondary animate-pulse";
    text.textContent = h.architecture ? `Runtime · ${h.architecture}` : "Backend online";
    if (h.architecture) $("#infra-arch").textContent = `IBM LinuxONE · ${h.architecture} · PyTorch CPU inference`;
    $("#infra-seg").textContent = h.segmentation_model || "—";
    $("#infra-inpaint").textContent = h.inpainting_model || "—";
    $("#p3-model").textContent = h.inpainting_model || "FLUX.1 Fill";
    $("#res-arch").textContent = h.architecture ? `· Runtime: ${h.architecture}` : "";
  }

  fetch("/api/health")
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error("health"))))
    .then((h) => { state.health = h; })
    .catch(() => { state.health = null; })
    .finally(renderRuntime);

  /* ───────────── Test Lab: staging a photo ───────────── */

  const dropzone = $("#dropzone");
  const fileInput = $("#file-input");
  const genBtn = $("#generate-btn");

  function setHint(icon, msg, cls) {
    const hint = $("#generate-hint");
    hint.className = `mt-space-xs font-label-md text-label-md tracking-wider flex items-center gap-1 ${cls}`;
    hint.firstElementChild.textContent = icon;
    hint.lastElementChild.textContent = msg;
  }

  function setFlow(active) {
    $$("#lab-flow .flow-step").forEach((el) => {
      const n = Number(el.dataset.step);
      el.dataset.state = n < active ? "done" : n === active ? "active" : "pending";
    });
  }

  function stageFile(file) {
    if (!file) return;
    if (!ALLOWED_TYPES.includes(file.type)) {
      showError("Please choose a JPG, PNG or WebP image.");
      return;
    }
    if (file.size > MAX_BYTES) {
      showError(`That image is ${fmtBytes(file.size)}; the maximum is 8 MB.`);
      return;
    }
    clearError();
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    state.file = file;
    state.previewUrl = URL.createObjectURL(file);

    $("#staged-preview").src = state.previewUrl;
    $("#staged-name").textContent = file.name;
    $("#staged-meta").textContent = `${fmtBytes(file.size)} · Ready for analysis`;
    $("#drop-empty").hidden = true;
    $("#drop-staged").hidden = false;

    genBtn.disabled = false;
    $("#generate-icon").textContent = "auto_awesome";
    setHint("check", "Photo staged. Ready to run the SegFormer analysis.", "text-secondary");
    setFlow(1);
  }

  function clearStaged() {
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    state.file = null;
    state.previewUrl = null;
    fileInput.value = "";
    $("#staged-preview").removeAttribute("src");
    $("#drop-empty").hidden = false;
    $("#drop-staged").hidden = true;
    genBtn.disabled = true;
    $("#generate-icon").textContent = "lock";
    setHint("info", "Select or drop a photo to begin", "text-outline");
    setFlow(1);
  }

  dropzone.addEventListener("click", (e) => {
    if (e.target.closest("#staged-clear")) return;
    fileInput.click();
  });
  dropzone.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && e.target === dropzone) {
      e.preventDefault();
      fileInput.click();
    }
  });
  dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.dataset.drag = "true"; });
  dropzone.addEventListener("dragleave", () => { delete dropzone.dataset.drag; });
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    delete dropzone.dataset.drag;
    stageFile(e.dataTransfer.files && e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", () => stageFile(fileInput.files && fileInput.files[0]));
  $("#staged-clear").addEventListener("click", (e) => { e.stopPropagation(); clearStaged(); });

  $("#load-sample").addEventListener("click", async () => {
    try {
      const res = await fetch("/assets/img/sample-park.jpg");
      if (!res.ok) throw new Error("not found");
      const blob = await res.blob();
      stageFile(new File([blob], "sample-park.jpg", { type: "image/jpeg" }));
      dropzone.scrollIntoView({ behavior: "smooth", block: "center" });
    } catch {
      showError("Couldn't load the sample photo. Please upload your own.");
    }
  });

  /* ───────────── Test Lab: run the pipeline ───────────── */

  function setBusy(busy) {
    state.busy = busy;
    genBtn.dataset.busy = String(busy);
    genBtn.setAttribute("aria-busy", String(busy));
    const icon = $("#generate-icon");
    icon.classList.toggle("animate-spin", busy);
    if (busy) {
      icon.textContent = "progress_activity";
      $("#generate-text").textContent = "Segmenting on IBM LinuxONE…";
      setHint("hourglass_top", "Running SegFormer-B2 on the CPU. This can take a little while.", "text-tertiary");
      setFlow(2);
    } else {
      icon.textContent = state.file ? "auto_awesome" : "lock";
      $("#generate-text").textContent = "Generate my illustration";
      if (state.file) setHint("check", "Photo staged. Ready to run the SegFormer analysis.", "text-secondary");
      setFlow(state.result ? 3 : 1);
    }
  }

  $("#upload-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!state.file || state.busy) return;
    clearError();
    stopPolling();
    setBusy(true);
    try {
      const body = new FormData();
      body.append("file", state.file);
      const res = await fetch("/api/process", { method: "POST", body });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(errorDetail(data, res));
      state.result = data;
      renderResults();
      location.hash = "#/results";
      if (data.job_url) startPolling(data.job_url);
      else failIllustration("The server did not return an illustration job.");
    } catch (err) {
      showError(err.message || "Failed to process the image.");
    } finally {
      setBusy(false);
    }
  });

  /* ───────────── Results ───────────── */

  const CLASSES = [
    { key: "water_body", label: "Water surface", swatch: "bg-cyan-400", bar: "bg-cyan-400", text: "text-cyan-300", chg: "chg-water" },
    { key: "arboreal_canopy", label: "Vegetation canopy", swatch: "bg-secondary", bar: "bg-secondary", text: "text-secondary", chg: "chg-canopy" },
    { key: "turf_ground", label: "Ground / turf", swatch: "bg-tertiary", bar: "bg-tertiary", text: "text-tertiary", chg: "chg-turf" },
  ];

  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  function renderMetrics(coverage) {
    const box = $("#p2-metrics");
    box.replaceChildren();
    CLASSES.forEach((c, i) => {
      const raw = Number(coverage && coverage[c.key]);
      const pct = Number.isFinite(raw) ? Math.min(100, Math.max(0, raw)) : 0;
      const row = el("div", `flex items-center justify-between text-body-sm${i ? " pt-1" : ""}`);
      const left = el("div", "flex items-center gap-2");
      left.append(el("span", `w-3 h-3 rounded ${c.swatch}`), el("span", "text-on-surface font-title-md", c.label));
      row.append(left, el("span", `font-headline-md text-body-md ${c.text} font-bold`, `${pct.toFixed(1)}%`));
      const track = el("div", "w-full h-1.5 rounded-full bg-surface-container-highest overflow-hidden");
      const fill = el("div", `h-full ${c.bar} rounded-full`);
      fill.style.width = `${pct}%`;
      track.append(fill);
      box.append(row, track);
      $(`#${c.chg}`).textContent = `${pct.toFixed(1)}% of scene`;
    });
  }

  function setP2(mode) {
    const imgs = (state.result && state.result.images) || {};
    if (mode === "overlay" && !imgs.overlay) mode = "mask";
    state.p2Mode = mode;
    $("#p2-img").src = imgs[mode] || imgs.mask || "";
    $$("[data-p2]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.p2 === mode)));
  }
  $$("[data-p2]").forEach((b) => b.addEventListener("click", () => setP2(b.dataset.p2)));

  const BADGES = {
    working: { box: "bg-tertiary/15 text-tertiary", dot: "bg-tertiary animate-pulse" },
    done: { box: "bg-secondary/15 text-secondary", dot: "bg-secondary" },
    error: { box: "bg-error-container/30 text-error", dot: "bg-error" },
  };
  function setBadge(kind, text) {
    const b = BADGES[kind];
    $("#res-badge").className = `inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full font-label-md text-label-md font-semibold tracking-wider uppercase ${b.box}`;
    $("#res-badge-dot").className = `w-2 h-2 rounded-full ${b.dot}`;
    $("#res-badge-text").textContent = text;
  }

  const STAGE_ICON = { done: "check", final: "check", active: "sync", pending: "more_horiz", error: "close" };
  function setStage(n, st) {
    const node = $(`#res-stages [data-stage="${n}"]`);
    node.dataset.state = st;
    const icon = $(".material-symbols-outlined", node);
    icon.textContent = STAGE_ICON[st];
    icon.classList.toggle("animate-spin", st === "active");
  }

  function setP3Waiting(spinning, title, sub) {
    $("#p3-img").hidden = true;
    $("#p3-wait").hidden = false;
    const spinner = $("#p3-spinner");
    spinner.textContent = spinning ? "progress_activity" : "error";
    spinner.classList.toggle("animate-spin", spinning);
    spinner.classList.toggle("text-tertiary", spinning);
    spinner.classList.toggle("text-error", !spinning);
    $("#p3-status").textContent = title;
    $("#p3-sub").textContent = sub;
  }

  function renderResults() {
    const r = state.result;
    $("#res-file").textContent = state.file ? state.file.name : "";
    $("#p1-img").src = r.images.input;
    $("#p1-orig").textContent = fmtSize(r.original_size);
    $("#p1-norm").textContent = fmtSize(r.normalized_size);
    setP2(state.p2Mode);
    renderMetrics(r.area_proportions);
    $("#res-prompt").textContent = r.dynamic_prompt || "";
    renderRuntime();

    $("#p3-open").hidden = true;
    $("#echo-a").src = r.images.input;
    $("#echo-b").hidden = true;
    $("#echo-b").removeAttribute("src");
    $("#echo-b-wait").hidden = false;
    setEchoEnabled(false);
    state.outputUrl = null;

    setStage(1, "done");
    setStage(2, "done");
    setStage(3, "active");
    setStage(4, "pending");
    setBadge("working", "Segmentation complete · illustrating");
    setP3Waiting(true, "Generating inpainting via FLUX.1 Fill…", "Starting · 0s");
    $('[data-nav="results"]').hidden = false;
  }

  /* ───────────── polling the illustration job ───────────── */

  function stopPolling() {
    clearInterval(state.poll);
    clearInterval(state.tick);
    state.poll = state.tick = null;
  }

  function startPolling(jobUrl) {
    stopPolling();
    state.jobUrl = jobUrl;
    state.startedAt = Date.now();
    let status = "starting";
    let inFlight = false;
    let errors = 0;
    const elapsed = () => Math.floor((Date.now() - state.startedAt) / 1000);
    const paint = () => {
      const label = status === "processing" ? "Processing" : "Starting";
      $("#p3-sub").textContent = `${label} · ${elapsed()}s`;
      $("#p3-status").textContent =
        status === "processing" ? "Generating inpainting via FLUX.1 Fill…" : "Queued on Replicate…";
    };
    paint();
    state.tick = setInterval(paint, 1000);

    const check = async () => {
      if (inFlight) return;
      if (Date.now() - state.startedAt > POLL_MAX_MS) {
        failIllustration("The illustration is taking too long. Please try again.");
        return;
      }
      inFlight = true;
      try {
        const res = await fetch(`/api/status?job_url=${encodeURIComponent(jobUrl)}`);
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(errorDetail(data, res));
        errors = 0;
        status = data.status || status;
        if (data.status === "succeeded" && data.output_url) {
          finishIllustration(data.output_url);
        } else if (data.status === "succeeded") {
          failIllustration("The job finished without returning an image.");
        } else if (data.status === "failed" || data.status === "canceled") {
          failIllustration(`Generation ${data.status}: ${data.error || "Please try again."}`);
        } else {
          paint();
        }
      } catch (err) {
        if (++errors >= POLL_MAX_ERRORS) failIllustration(err.message || "Failed to check generation status.");
      } finally {
        inFlight = false;
      }
    };
    check();
    state.poll = setInterval(check, POLL_MS);
  }

  function finishIllustration(url) {
    stopPolling();
    if (!/^https:\/\//i.test(url)) {
      failIllustration("The illustration URL was not valid.");
      return;
    }
    state.outputUrl = url;
    const img = $("#p3-img");
    img.src = url;
    img.hidden = false;
    $("#p3-wait").hidden = true;
    const open = $("#p3-open");
    open.href = url;
    open.hidden = false;
    $("#echo-b").src = url;
    $("#echo-b").hidden = false;
    $("#echo-b-wait").hidden = true;
    setStage(3, "done");
    setStage(4, "final");
    setBadge("done", "Analysis complete");
    setEchoEnabled(true);
    toast("Illustration ready.");
  }

  function failIllustration(msg) {
    stopPolling();
    setStage(3, "error");
    setStage(4, "pending");
    setBadge("error", "Illustration failed");
    setP3Waiting(false, "Illustration unavailable", msg);
    showError(msg);
  }

  $("#res-new").addEventListener("click", () => {
    stopPolling();
    state.result = null;
    state.outputUrl = null;
    clearStaged();
    location.hash = "#/lab";
  });

  /* ───────────── Echo card ───────────── */

  function setEchoEnabled(on) {
    $("#echo-download").disabled = !on;
    $("#echo-share").disabled = !on;
    $("#echo-hint").textContent = on
      ? "Export a PNG sized for your format."
      : "Available once the illustration is ready.";
  }

  $$("#echo-format [data-format]").forEach((b) =>
    b.addEventListener("click", () => {
      state.format = b.dataset.format;
      $("#echo-card").dataset.format = state.format;
      $$("#echo-format [data-format]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    })
  );

  const CARD_SIZES = { square: [1080, 1080], story: [1080, 1920], landscape: [1200, 628] };
  const C = { bg: "#02110c", panel: "#11231d", line: "#263832", text: "#d2e7dd", dim: "#c1c8c3", muted: "#8b928e", green: "#63dcaa", gold: "#f4be63", goldDeep: "#462e00", goldLight: "#ffdeac" };

  function loadImg(src) {
    return new Promise((resolve, reject) => {
      const im = new Image();
      im.onload = () => resolve(im);
      im.onerror = reject;
      im.src = src;
    });
  }

  // The illustration lives on another origin; fetch it as a blob so the canvas
  // isn't tainted. If the host blocks that, the card is exported without it.
  async function loadRemote(url) {
    try {
      const res = await fetch(url, { mode: "cors" });
      if (!res.ok) return null;
      const objectUrl = URL.createObjectURL(await res.blob());
      try { return await loadImg(objectUrl); } finally { URL.revokeObjectURL(objectUrl); }
    } catch {
      return null;
    }
  }

  function drawCover(ctx, im, x, y, w, h) {
    const s = Math.max(w / im.width, h / im.height);
    const sw = w / s, sh = h / s;
    ctx.save();
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, 12);
    ctx.clip();
    ctx.drawImage(im, (im.width - sw) / 2, (im.height - sh) / 2, sw, sh, x, y, w, h);
    ctx.restore();
  }

  function chip(ctx, text, x, y, fg, bg) {
    ctx.font = "600 15px Inter, sans-serif";
    const w = ctx.measureText(text).width + 24;
    ctx.fillStyle = bg;
    ctx.beginPath();
    ctx.roundRect(x, y, w, 30, 6);
    ctx.fill();
    ctx.fillStyle = fg;
    ctx.textBaseline = "middle";
    ctx.fillText(text, x + 12, y + 16);
    ctx.textBaseline = "alphabetic";
  }

  function wrap(ctx, text, x, y, maxW, lineH) {
    let line = "";
    for (const word of text.split(" ")) {
      const test = line ? `${line} ${word}` : word;
      if (ctx.measureText(test).width > maxW && line) {
        ctx.fillText(line, x, y);
        line = word;
        y += lineH;
      } else {
        line = test;
      }
    }
    if (line) { ctx.fillText(line, x, y); y += lineH; }
    return y;
  }

  async function renderEchoCanvas(format) {
    try {
      await Promise.all([
        document.fonts.load("700 40px Syne"),
        document.fonts.load("500 20px 'Plus Jakarta Sans'"),
        document.fonts.load("600 15px Inter"),
      ]);
    } catch { /* fall back to system fonts */ }

    const [W, H] = CARD_SIZES[format];
    const canvas = document.createElement("canvas");
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext("2d");
    const wide = format === "landscape";
    const P = wide ? 36 : 56;

    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = C.line;
    ctx.lineWidth = 2;
    ctx.strokeRect(1, 1, W - 2, H - 2);

    const [mark, photo, art] = await Promise.all([
      loadImg("/assets/img/mark.png").catch(() => null),
      loadImg(state.result.images.input).catch(() => null),
      loadRemote(state.outputUrl),
    ]);

    // header
    const hs = wide ? 36 : 52;
    if (mark) ctx.drawImage(mark, P, P, hs, hs);
    ctx.fillStyle = C.text;
    ctx.font = `700 ${wide ? 26 : 36}px Syne, sans-serif`;
    ctx.fillText("MineScars", P + hs + 14, P + hs * 0.72);

    // images
    const gap = 16;
    const imgTop = P + hs + (wide ? 18 : 30);
    let slots, textX, textW, textY;
    if (format === "story") {
      const w = W - 2 * P, h = 560;
      slots = [[P, imgTop, w, h], [P, imgTop + h + gap, w, h]];
      textX = P; textW = w; textY = imgTop + 2 * h + gap + 80;
    } else if (wide) {
      const colW = 700, w = (colW - gap) / 2, h = H - imgTop - P - 40;
      slots = [[P, imgTop, w, h], [P + w + gap, imgTop, w, h]];
      textX = P + colW + 32; textW = W - textX - P; textY = imgTop + 40;
    } else {
      const w = (W - 2 * P - gap) / 2, h = 420;
      slots = [[P, imgTop, w, h], [P + w + gap, imgTop, w, h]];
      textX = P; textW = W - 2 * P; textY = imgTop + h + 84;
    }

    ctx.fillStyle = C.panel;
    slots.forEach(([x, y, w, h]) => { ctx.beginPath(); ctx.roundRect(x - 4, y - 4, w + 8, h + 8, 14); ctx.fill(); });
    if (photo) drawCover(ctx, photo, ...slots[0]);
    chip(ctx, "YOUR PLACE", slots[0][0] + 14, slots[0][1] + slots[0][3] - 44, C.text, "rgba(2,17,12,0.8)");
    if (art) {
      drawCover(ctx, art, ...slots[1]);
    } else {
      const [x, y, w, h] = slots[1];
      ctx.fillStyle = C.bg;
      ctx.fillRect(x, y, w, h);
      ctx.fillStyle = C.muted;
      ctx.font = "500 18px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Illustration unavailable for export", x + w / 2, y + h / 2);
      ctx.textAlign = "left";
    }
    chip(ctx, "ILLUSTRATIVE SCAR", slots[1][0] + 14, slots[1][1] + slots[1][3] - 44, C.goldLight, C.goldDeep);

    // copy
    let y = textY;
    ctx.fillStyle = C.text;
    ctx.font = `700 ${wide ? 34 : 54}px Syne, sans-serif`;
    y = wrap(ctx, "“Your place. Their scars.”", textX, y, textW, wide ? 42 : 64);
    y += 6;
    ctx.fillStyle = C.green;
    ctx.font = `600 ${wide ? 17 : 24}px 'Plus Jakarta Sans', sans-serif`;
    y = wrap(ctx, "La Pampa, Madre de Dios, Peru · Documented by MAAP 2025", textX, y, textW, wide ? 24 : 34);
    y += 8;
    ctx.fillStyle = C.dim;
    ctx.font = `400 ${wide ? 15 : 22}px 'Plus Jakarta Sans', sans-serif`;
    wrap(
      ctx,
      "Over 139,000 hectares of biodiverse rainforest stripped for alluvial gold. This simulation illustrates the vulnerability of our green sanctuaries.",
      textX, y, textW, wide ? 22 : 32
    );

    ctx.fillStyle = C.muted;
    ctx.font = `400 ${wide ? 12 : 17}px 'Plus Jakarta Sans', sans-serif`;
    wrap(ctx, "AI illustration based on MAAP satellite evidence. Not an actual forecast of this location.", P, H - P + (wide ? -8 : -6), W - 2 * P, 22);

    return { canvas, hadArt: Boolean(art) };
  }

  const canvasBlob = (canvas) => new Promise((res) => canvas.toBlob(res, "image/png"));

  async function buildEchoFile() {
    const { canvas, hadArt } = await renderEchoCanvas(state.format);
    const blob = await canvasBlob(canvas);
    if (!blob) throw new Error("Couldn't render the Echo Card.");
    return { file: new File([blob], `minescars-echo-${state.format}.png`, { type: "image/png" }), hadArt };
  }

  function withBusy(btn, fn) {
    return async () => {
      if (btn.disabled || !state.result || !state.outputUrl) return;
      btn.disabled = true;
      try { await fn(); }
      catch (err) { if (!err || err.name !== "AbortError") showError(err.message || "Something went wrong."); }
      finally { btn.disabled = !state.outputUrl; }
    };
  }

  $("#echo-download").addEventListener("click", withBusy($("#echo-download"), async () => {
    const { file, hadArt } = await buildEchoFile();
    const url = URL.createObjectURL(file);
    const a = document.createElement("a");
    a.href = url;
    a.download = file.name;
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    toast(hadArt ? "Echo Card downloaded." : "Echo Card downloaded (without the illustration — the image host blocked export).");
  }));

  $("#echo-share").addEventListener("click", withBusy($("#echo-share"), async () => {
    const { file } = await buildEchoFile();
    const data = { files: [file], title: "MineScars Echo Card", text: "Your place. Their scars." };
    if (navigator.canShare && navigator.canShare(data)) {
      await navigator.share(data);
    } else {
      const url = URL.createObjectURL(file);
      const a = document.createElement("a");
      a.href = url;
      a.download = file.name;
      document.body.append(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      toast("Sharing isn't supported on this device, so the card was downloaded.");
    }
  }));

  /* ───────────── boot ───────────── */

  setFlow(1);
  route();
})();
