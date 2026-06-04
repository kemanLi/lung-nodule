(function () {
  const wrap = document.getElementById("viewer-wrap");
  if (!wrap) return;

  const img = document.getElementById("viewer-img");
  const canvas = document.getElementById("viewer-canvas");
  const hintEl = document.getElementById("viewer-mode-hint");
  const ctx = canvas.getContext("2d");
  const detections = JSON.parse(wrap.dataset.detections || "[]");
  const overlayUrl = wrap.dataset.overlay;
  const inputUrl = wrap.dataset.input;
  let selected = null;

  /* 与 pipeline 叠加图里的青黄框 (0,220,255) 错开，用细角标 + 洋红/青绿系 */
  const STYLES = {
    high: { stroke: "#E040FB", width: 1.25 },
    mid: { stroke: "#69F0AE", width: 1.25 },
    low: { stroke: "#FFEE58", width: 1 },
    selected: { stroke: "#FFFFFF", width: 1.75 },
  };

  function currentBase() {
    return document.querySelector('input[name="base"]:checked')?.value || "input";
  }

  function levelFor(conf) {
    const c = Number(conf) || 0;
    if (c >= 0.6) return "high";
    if (c >= 0.4) return "mid";
    return "low";
  }

  function styleFor(i, conf) {
    if (i === selected) return STYLES.selected;
    return STYLES[levelFor(conf)];
  }

  function shouldDrawBoxes() {
    const show = document.getElementById("show-boxes");
    return show && show.checked && currentBase() === "input";
  }

  function updateHint() {
    if (!hintEl) return;
    const base = currentBase();
    const show = document.getElementById("show-boxes")?.checked;
    if (base === "overlay") {
      hintEl.textContent = "当前为 AI 叠加图（已含内置框线与分割），交互检测框已隐藏以免重复。";
      hintEl.hidden = false;
    } else if (show) {
      hintEl.textContent = "原图上的细角标为交互检测框，与 AI 叠加图内置标注分开显示。";
      hintEl.hidden = false;
    } else {
      hintEl.hidden = true;
    }
  }

  function syncCanvasSize() {
    const w = img.clientWidth;
    const h = img.clientHeight;
    canvas.width = w;
    canvas.height = h;
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";
    drawBoxes();
  }

  /** 四角 L 形细线，不铺填充，避免与叠加图糊成一团 */
  function drawCornerBrackets(x, y, w, h, style) {
    const len = Math.max(6, Math.min(18, w * 0.22, h * 0.22));
    ctx.save();
    ctx.strokeStyle = style.stroke;
    ctx.lineWidth = style.width;
    ctx.lineCap = "square";
    if (selected !== null && style === STYLES.selected) {
      ctx.setLineDash([]);
      ctx.strokeStyle = "#E040FB";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
    } else {
      ctx.setLineDash([4, 3]);
    }
    const drawCorner = (x0, y0, dx, dy) => {
      ctx.beginPath();
      ctx.moveTo(x0, y0 + dy * len);
      ctx.lineTo(x0, y0);
      ctx.lineTo(x0 + dx * len, y0);
      ctx.stroke();
    };
    drawCorner(x, y, 1, 1);
    drawCorner(x + w, y, -1, 1);
    drawCorner(x, y + h, 1, -1);
    drawCorner(x + w, y + h, -1, -1);
    ctx.restore();
  }

  function drawLabel(x, y, text, style) {
    const padX = 4;
    const padY = 2;
    ctx.font = "11px sans-serif";
    const tw = ctx.measureText(text).width;
    const lx = x;
    const ly = Math.max(0, y - 14);
    ctx.fillStyle = "rgba(0, 0, 0, 0.72)";
    ctx.fillRect(lx, ly, tw + padX * 2, 14);
    ctx.fillStyle = style.stroke;
    ctx.fillText(text, lx + padX, ly + 11);
  }

  function drawBoxes() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    updateHint();

    if (!shouldDrawBoxes() || !img.naturalWidth) return;

    const sx = canvas.width / img.naturalWidth;
    const sy = canvas.height / img.naturalHeight;

    detections.forEach((d, i) => {
      const x = d.x1 * sx;
      const y = d.y1 * sy;
      const w = (d.x2 - d.x1) * sx;
      const h = (d.y2 - d.y1) * sy;
      const style = styleFor(i, d.confidence);
      drawCornerBrackets(x, y, w, h, style);
      const label = "#" + (i + 1) + " " + ((d.confidence || 0) * 100).toFixed(0) + "%";
      drawLabel(x, y, label, style);
    });
  }

  img.addEventListener("load", syncCanvasSize);
  window.addEventListener("resize", syncCanvasSize);
  if (img.complete) syncCanvasSize();

  document.querySelectorAll('input[name="base"]').forEach((radio) => {
    radio.addEventListener("change", () => {
      img.src = radio.value === "input" ? inputUrl : overlayUrl;
      updateHint();
    });
  });

  const showBoxes = document.getElementById("show-boxes");
  if (showBoxes) {
    showBoxes.addEventListener("change", () => {
      drawBoxes();
      updateHint();
    });
  }

  document.querySelectorAll("#nodule-list .nodule-item").forEach((li) => {
    li.addEventListener("click", () => {
      selected = parseInt(li.dataset.index, 10);
      document.querySelectorAll("#nodule-list .nodule-item").forEach((el) => el.classList.remove("active"));
      li.classList.add("active");
      if (currentBase() !== "input") {
        const inputRadio = document.querySelector('input[name="base"][value="input"]');
        if (inputRadio) {
          inputRadio.checked = true;
          img.src = inputUrl;
        }
      }
      li.scrollIntoView({ block: "nearest", behavior: "smooth" });
      drawBoxes();
      updateHint();
    });
  });

  updateHint();
})();
