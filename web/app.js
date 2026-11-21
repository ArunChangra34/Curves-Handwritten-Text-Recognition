/* Curves — in-browser handwriting recognition (CRNN, ONNX Runtime Web) */
const H = 32, W = 160;
const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
const state = { script: "auto", session: null, cs: null, busy: false };

/* ---------- theme ---------- */
const ICON = {
  sun: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>',
  moon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>'
};
function paintTheme() { $("#theme").innerHTML = document.documentElement.dataset.theme === "dark" ? ICON.sun : ICON.moon; clearPad(); }
$("#theme").onclick = () => {
  const t = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = t; try { localStorage.setItem("curves-theme", t); } catch (e) {} paintTheme();
};

/* ---------- routing ---------- */
function route() {
  const r = (location.hash.replace("#/", "") || "recognize"); const name = ["recognize", "model", "pipeline"].includes(r) ? r : "recognize";
  $$(".view").forEach(v => v.hidden = v.id !== "view-" + name);
  $$(".links a").forEach(a => a.classList.toggle("on", a.dataset.route === name));
  window.scrollTo(0, 0);
}
addEventListener("hashchange", route);

/* ---------- preprocessing (mirror of ml/imgproc.py) ---------- */
function toGray(img, maxSide = 1024) {
  const s = Math.min(1, maxSide / Math.max(img.width, img.height));
  const w = Math.max(1, Math.round(img.width * s)), h = Math.max(1, Math.round(img.height * s));
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  const x = c.getContext("2d", { willReadFrequently: true }); x.fillStyle = "#fff"; x.fillRect(0, 0, w, h); x.drawImage(img, 0, 0, w, h);
  const d = x.getImageData(0, 0, w, h).data, g = new Float32Array(w * h);
  for (let i = 0; i < g.length; i++) g[i] = 0.299 * d[4 * i] + 0.587 * d[4 * i + 1] + 0.114 * d[4 * i + 2];
  return { g, w, h };
}
function percentile(g, q) {
  const hist = new Uint32Array(256); for (let i = 0; i < g.length; i++) hist[Math.min(255, Math.max(0, Math.round(g[i])))]++;
  const t = q * (g.length - 1); let acc = 0; for (let v = 0; v < 256; v++) { acc += hist[v]; if (acc > t) return v; } return 255;
}
function gauss1(a, w, h, sigma = 1) {
  const r = 3, k = []; let s = 0; for (let i = -r; i <= r; i++) { const v = Math.exp(-i * i / (2 * sigma * sigma)); k.push(v); s += v; } for (let i = 0; i < k.length; i++) k[i] /= s;
  const t = new Float32Array(a.length), o = new Float32Array(a.length), cl = (v, m) => v < 0 ? 0 : v >= m ? m - 1 : v;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { let v = 0; for (let i = -r; i <= r; i++) v += k[i + r] * a[y * w + cl(x + i, w)]; t[y * w + x] = v; }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { let v = 0; for (let i = -r; i <= r; i++) v += k[i + r] * t[cl(y + i, h) * w + x]; o[y * w + x] = v; }
  return o;
}
function preprocess(img) {
  let { g, w, h } = toGray(img);
  if (percentile(g, 0.5) < 128) for (let i = 0; i < g.length; i++) g[i] = 255 - g[i];      // median < 128 -> flip
  const bg = percentile(g, 0.5), dark = percentile(g, 0.005), span = Math.max(bg - dark, 25);
  const ink = new Float32Array(g.length);
  for (let i = 0; i < g.length; i++) { const v = Math.min(1, Math.max(0, (bg - g[i]) / span)); ink[i] = v < 0.15 ? 0 : v; }
  const m = gauss1(ink, w, h); let x0 = w, x1 = -1, y0 = h, y1 = -1;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (m[y * w + x] > 0.3) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  if (x1 < 0) return null;                                                                // nothing written
  const pad = 2; x0 = Math.max(x0 - pad, 0); y0 = Math.max(y0 - pad, 0); x1 = Math.min(x1 + pad, w - 1); y1 = Math.min(y1 + pad, h - 1);
  const cw = x1 - x0 + 1, ch = y1 - y0 + 1, src = document.createElement("canvas"); src.width = cw; src.height = ch;
  const sx = src.getContext("2d"), id = sx.createImageData(cw, ch);
  for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) { const v = ink[(y + y0) * w + x + x0] * 255, j = 4 * (y * cw + x); id.data[j] = id.data[j + 1] = id.data[j + 2] = v; id.data[j + 3] = 255; }
  sx.putImageData(id, 0, 0);
  const nh = H - 4, nw = Math.max(1, Math.min(W - 8, Math.round(cw * nh / ch)));
  const out = document.createElement("canvas"); out.width = W; out.height = H; const ox = out.getContext("2d", { willReadFrequently: true });
  ox.fillStyle = "#000"; ox.fillRect(0, 0, W, H); ox.imageSmoothingEnabled = true; ox.imageSmoothingQuality = "high"; ox.drawImage(src, 0, 0, cw, ch, 4, 2, nw, nh);
  const d = ox.getImageData(0, 0, W, H).data, f = new Float32Array(W * H); for (let i = 0; i < f.length; i++) f[i] = d[4 * i] / 255;
  return { data: f, canvas: out };
}

/* ---------- inference ---------- */
function decode(logits, T, C, mask) {
  let prev = -1, s = "", conf = [];
  for (let t = 0; t < T; t++) {
    let mx = -1e9, k = 0, den = 0; const o = t * C;
    for (let c = 0; c < C; c++) if (mask[c]) { if (logits[o + c] > mx) { mx = logits[o + c]; k = c; } }
    for (let c = 0; c < C; c++) if (mask[c]) den += Math.exp(logits[o + c] - mx);
    if (k !== prev && k !== 0) { s += state.cs.chars[k - 1]; conf.push(1 / den); }
    prev = k;
  }
  return { text: s, conf: conf.length ? conf.reduce((a, b) => a + b) / conf.length : 0 };
}
async function recognize(img) {
  if (!state.session || state.busy) return; state.busy = true;
  try {
    const p = preprocess(img), box = $("#result"), out = $("#out"); box.hidden = false;
    if (!p) { out.textContent = "No handwriting detected"; out.className = "text empty"; ["#m-conf", "#m-script", "#m-ms"].forEach(s => $(s).textContent = "–"); return; }
    $("#prev").getContext("2d").drawImage(p.canvas, 0, 0);
    const t0 = performance.now(), res = await state.session.run({ image: new ort.Tensor("float32", p.data, [1, 1, H, W]) });
    const L = res.logits, r = decode(L.data, L.dims[1], L.dims[2], state.cs.masks[state.script]), ms = performance.now() - t0;
    const dv = [...r.text].filter(c => c >= "ऀ" && c <= "ॿ").length, la = [...r.text].filter(c => /[A-Za-z0-9]/.test(c)).length;
    out.className = "text"; out.textContent = r.text || "—"; out.lang = dv > la ? "hi" : "en";
    $("#m-conf").textContent = r.text ? (100 * r.conf).toFixed(1) + "%" : "–";
    $("#m-script").textContent = !r.text ? "–" : dv && la ? "Mixed" : dv ? "Devanagari" : "English";
    $("#m-ms").textContent = ms.toFixed(0) + " ms";
  } catch (e) { console.error(e); $("#out").textContent = "Recognition failed"; $("#out").className = "text empty"; $("#result").hidden = false; }
  finally { state.busy = false; }
}
function fromFile(f) { if (!f || !f.type.startsWith("image/")) return; const im = new Image(); im.onload = () => { recognize(im); URL.revokeObjectURL(im.src); }; im.src = URL.createObjectURL(f); }

/* ---------- input widgets ---------- */
const drop = $("#drop"), file = $("#file");
file.onchange = () => { fromFile(file.files[0]); file.value = ""; };
drop.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); file.click(); } };
["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => fromFile(e.dataTransfer.files[0]));
addEventListener("paste", e => { const it = [...(e.clipboardData?.items || [])].find(i => i.type.startsWith("image/")); if (it) fromFile(it.getAsFile()); });
$$(".tab").forEach(b => b.onclick = () => {
  $$(".tab").forEach(x => x.classList.toggle("on", x === b)); $("#pane-drop").hidden = b.dataset.tab !== "drop"; $("#pane-draw").hidden = b.dataset.tab !== "draw"; if (b.dataset.tab === "draw") clearPad();
});
$$("#script .chip").forEach(b => b.onclick = () => { state.script = b.dataset.s; $$("#script .chip").forEach(x => x.classList.toggle("on", x === b)); if (lastImg) recognize(lastImg); });

/* drawing pad */
const pad = $("#pad"), px = pad.getContext("2d", { willReadFrequently: true }); let drawing = false, timer = 0, lastImg = null, dirty = false;
function css(v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }
function clearPad() { px.fillStyle = css("--card") || "#fff"; px.fillRect(0, 0, pad.width, pad.height); dirty = false; }
function pt(e) { const r = pad.getBoundingClientRect(); return [(e.clientX - r.left) * pad.width / r.width, (e.clientY - r.top) * pad.height / r.height]; }
pad.onpointerdown = e => { drawing = true; pad.setPointerCapture(e.pointerId); clearTimeout(timer); const [x, y] = pt(e); px.beginPath(); px.moveTo(x, y); px.lineTo(x + .1, y); px.strokeStyle = css("--text"); px.lineWidth = 10; px.lineCap = px.lineJoin = "round"; px.stroke(); dirty = true; };
pad.onpointermove = e => { if (!drawing) return; const [x, y] = pt(e); px.lineTo(x, y); px.stroke(); };
pad.onpointerup = pad.onpointercancel = () => { if (!drawing) return; drawing = false; timer = setTimeout(() => { const c = document.createElement("canvas"); c.width = pad.width; c.height = pad.height; c.getContext("2d").drawImage(pad, 0, 0); lastImg = c; recognize(c); }, 450); };
$("#clear").onclick = () => { clearPad(); $("#result").hidden = true; lastImg = null; };
$("#copy").onclick = async () => { const b = $("#copy"); try { await navigator.clipboard.writeText($("#out").textContent); b.textContent = "COPIED"; setTimeout(() => b.textContent = "COPY", 1200); } catch (e) {} };
/* remember last image so changing script re-runs it */
const _rec = recognize; recognize = async function (img) { lastImg = img; return _rec(img); };

/* samples */
const SAMPLES = [["notebook", "notebook"], ["Forest 24", "forest24"], ["किताब", "kitab"], ["भारत", "bharat"], ["hello नमस्ते", "mixed"]];
SAMPLES.forEach(([label, f]) => { const b = document.createElement("button"); b.className = "chip"; b.textContent = label; b.lang = /[ऀ-ॿ]/.test(label) ? "hi" : "en"; b.style.fontFamily = "var(--dev)"; b.style.letterSpacing = "0";
  b.onclick = () => { const im = new Image(); im.onload = () => { $$(".tab")[0].click(); recognize(im); }; im.src = `samples/${f}.png`; }; $("#samples").append(b); });

/* ---------- model + pipeline pages ---------- */
const FLOW = [["Image", "32 × 160 · grayscale"], ["CNN × 6", "conv · BN · ReLU · pool", 1], ["Sequence", "40 steps × 256"], ["BiLSTM × 2", "hidden 128 / direction", 1], ["Linear", "per-step class scores"], ["CTC decode", "collapse repeats + blanks", 1]];
$("#flow").innerHTML = FLOW.map(([a, b, h]) => `<div class="node ${h ? "hl" : ""}"><b>${a}</b><span>${b}</span></div>`).join("");
const PHASES = [["Define the task", "Word and short-phrase recognition for Devanagari and English with a single network."], ["Build the character set", "One unified alphabet: Latin letters, digits, punctuation, Devanagari vowels, consonants, matras, halant and digits, plus the CTC blank."], ["Prepare the data", "Handwriting-style word images with slant, wobble, pen-thickness, blur and noise; held-out fonts keep the test honest."], ["Preprocess", "Greyscale, background estimate, ink cut-out, crop to the writing, resize to 32 × 160 — identical in training and in the browser."], ["Train", "CNN + BiLSTM with CTC loss, AdamW and a one-cycle learning-rate schedule."], ["Compare", "Against a plain CNN baseline, and against separate single-script models."], ["Export", "ONNX graph, executed locally with ONNX Runtime Web."], ["Deploy", "A static site on Vercel — no server, no upload."]];
$("#phases").innerHTML = PHASES.map(([a, b]) => `<li><b>${a}</b><span>${b}</span></li>`).join("");
async function loadResults() {
  try {
    const R = (await (await fetch("model/results.json")).json()).metrics, N = { crnn_joint: "CRNN · joint (en + hi)", cnn_baseline: "Plain CNN baseline", crnn_en: "CRNN · English only", crnn_hi: "CRNN · Devanagari only" };
    const cols = [["test_seen/all", "Seen fonts"], ["test_held/en", "Held-out · EN"], ["test_held/hi", "Held-out · HI"], ["test_held/mix", "Held-out · Mixed"], ["test_held/all", "Held-out · All"]];
    const best = {}; cols.forEach(([k]) => best[k] = Math.min(...Object.values(R).filter(r => r[k]).map(r => r[k].cer)));
    $("#tbl").innerHTML = `<tr><th>MODEL</th>${cols.map(c => `<th>${c[1].toUpperCase()}</th>`).join("")}<th>PARAMS</th></tr>` +
      Object.entries(R).map(([k, r]) => `<tr class="${r["test_held/all"] && r["test_held/all"].cer === best["test_held/all"] ? "best" : ""}"><td>${N[k] || k}</td>${cols.map(([c]) => `<td>${r[c] ? r[c].cer.toFixed(1) + "%" : "–"}</td>`).join("")}<td>${(r.params / 1e6).toFixed(2)}M</td></tr>`).join("");
    $("#tbl-note").textContent = "Character error rate on 6,000 unseen samples per set. “Held-out” samples use fonts the network never saw during training. “–” means the model cannot read that script.";
  } catch (e) { $("#tbl").innerHTML = ""; $("#tbl-note").textContent = "Evaluation results will appear here once the model has been evaluated."; }
}

/* ---------- boot ---------- */
(async function boot() {
  paintTheme(); route(); loadResults();
  try {
    state.cs = await (await fetch("model/charset.json")).json();
    ort.env.wasm.wasmPaths = new URL("vendor/", location.href).href; ort.env.wasm.numThreads = 1;
    state.session = await ort.InferenceSession.create("model/curves.onnx", { executionProviders: ["wasm"] });
    $("#dot").className = "dot ok"; $("#status-text").textContent = "Model ready";
  } catch (e) { console.error(e); $("#dot").className = "dot err"; $("#status-text").textContent = "Model failed to load"; }
})();
