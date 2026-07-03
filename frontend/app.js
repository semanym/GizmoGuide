const compareForm = document.querySelector("#compare-form");
const chatForm = document.querySelector("#chat-form");
const productA = document.querySelector("#product-a");
const productB = document.querySelector("#product-b");
const productAResults = document.querySelector("#product-a-results");
const productBResults = document.querySelector("#product-b-results");
const userMessage = document.querySelector("#user-message");
const submitButton = document.querySelector("#submit-button");
const chatStream = document.querySelector("#chat-stream");
const welcomeState = document.querySelector("#welcome-state");

const pickers = new Map();
let activeCandidates = [];
let hasCompared = false;
let sessionId = `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;

setupPicker(productA, productAResults);
setupPicker(productB, productBResults);

for (const button of document.querySelectorAll(".preset")) {
  button.addEventListener("click", () => {
    selectProduct(productA, button.dataset.a);
    selectProduct(productB, button.dataset.b);
  });
}

compareForm.addEventListener("submit", (event) => {
  event.preventDefault();
  startComparison();
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  await sendChatMessage();
});

userMessage.addEventListener("input", autosizeTextarea);

async function loadProducts() {
  await selectProduct(productA, "iPhone 15");
  await selectProduct(productB, "vivo X100");
}

function setupPicker(input, resultsNode) {
  const priceNode = document.createElement("div");
  priceNode.className = "picker-price";
  input.closest(".search-picker")?.appendChild(priceNode);

  const state = {
    input,
    resultsNode,
    priceNode,
    selected: null,
    searchTimer: null,
    requestSeq: 0,
  };
  pickers.set(input, state);

  input.addEventListener("input", () => {
    state.selected = null;
    setPickerPrice(state, "");
    clearTimeout(state.searchTimer);
    state.searchTimer = setTimeout(() => searchProducts(state, input.value.trim()), 220);
  });
  input.addEventListener("focus", () => {
    if (!state.selected && input.value.trim()) searchProducts(state, input.value.trim());
  });
}

document.addEventListener("click", (event) => {
  for (const state of pickers.values()) {
    if (!state.input.contains(event.target) && !state.resultsNode.contains(event.target)) {
      hideResults(state);
    }
  }
});

async function searchProducts(state, query) {
  if (!query) {
    hideResults(state);
    return;
  }
  const seq = ++state.requestSeq;
  try {
    const response = await fetch(`/products?q=${encodeURIComponent(query)}&limit=8`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (seq !== state.requestSeq) return;
    renderSearchResults(state, data.items || []);
  } catch (error) {
    renderSearchError(state, error);
  }
}

function renderSearchResults(state, items) {
  if (!items.length) {
    state.resultsNode.innerHTML = `<div class="search-empty">没有找到匹配机型</div>`;
    state.resultsNode.classList.remove("hidden");
    return;
  }
  state.resultsNode.innerHTML = items.map((item) => renderSearchItem(item)).join("");
  for (const option of state.resultsNode.querySelectorAll(".search-option")) {
    // 用 mousedown + preventDefault：避免点击 option 时 <label> 把焦点重新丢回 input，
    // 触发 focus 里的二次搜索导致下拉「回弹」。
    option.addEventListener("mousedown", (event) => {
      event.preventDefault();
      const item = items.find((candidate) => candidate.id === option.dataset.id);
      if (item) chooseProduct(state, item);
    });
  }
  state.resultsNode.classList.remove("hidden");
}

function renderSearchItem(item) {
  const price = getProductPrice(item);
  const meta = buildProductMetaLine(item, { skipPrice: true });
  return `
    <button type="button" class="search-option" data-id="${escapeHtml(item.id)}">
      <div class="search-option-head">
        <strong>${escapeHtml(item.name)}</strong>
        ${price ? `<b class="search-price">${escapeHtml(price)}</b>` : ""}
      </div>
      <span>${escapeHtml(meta || "ZOL 参数库")}</span>
    </button>
  `;
}

function renderSearchError(state, error) {
  state.resultsNode.innerHTML = `<div class="search-empty">${escapeHtml(error.message || "搜索失败")}</div>`;
  state.resultsNode.classList.remove("hidden");
}

function chooseProduct(state, item) {
  state.selected = item;
  state.input.value = item.name;
  setPickerPrice(state, getProductPrice(item));
  hideResults(state);
}

function setPickerPrice(state, price) {
  if (!state.priceNode) return;
  // 价格行始终占位（见 .picker-price 固定高度），只切换文字，避免选中后布局高度变化
  // 导致页面回弹、两列不对齐。
  state.priceNode.textContent = price ? `参考价 ${price}` : "";
}

function hideResults(state) {
  state.resultsNode.classList.add("hidden");
}

async function selectProduct(input, nameOrId) {
  const state = pickers.get(input);
  if (!state) return;
  const response = await fetch(`/products?q=${encodeURIComponent(nameOrId)}&limit=1`);
  if (!response.ok) return;
  const data = await response.json();
  const item = (data.items || [])[0];
  if (item) chooseProduct(state, item);
}

function startComparison() {
  const left = getSelectedProduct(productA);
  const right = getSelectedProduct(productB);
  if (!left || !right) {
    appendErrorMessage(new Error("请选择两款候选商品。"));
    return;
  }
  if (left.name === right.name) {
    appendErrorMessage(new Error("请选择两款不同的候选商品。"));
    return;
  }

  activeCandidates = [left.name, right.name];
  hasCompared = true;
  sessionId = `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  welcomeState.classList.add("hidden");
  chatForm.classList.remove("hidden");
  appendAssistantComparison(left, right);
  userMessage.focus();
  scrollToLatest();
}

async function sendChatMessage() {
  const message = userMessage.value.trim();
  if (!message || !hasCompared || activeCandidates.length < 2) return;

  appendUserMessage(message);
  userMessage.value = "";
  autosizeTextarea();

  const streamNode = appendStreamingMessage();
  setBusy(true);

  try {
    const payload = {
      session_id: sessionId,
      message,
      candidate_products: activeCandidates,
    };
    const data = await sendChatMessageStream(payload, streamNode);
    renderFinalAssistantResponse(streamNode, data);
  } catch (error) {
    renderStreamError(streamNode, error);
  } finally {
    setBusy(false);
  }
}

async function sendChatMessageStream(payload, streamNode) {
  const response = await fetch("/chat/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Accept": "text/event-stream",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  if (!response.body) return sendChatMessageFallback(payload);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";

    for (const rawEvent of events) {
      const parsed = parseSseEvent(rawEvent);
      if (!parsed) continue;
      const finalResponse = handleStreamEvent(parsed, streamNode);
      if (finalResponse) return finalResponse;
    }
  }

  if (buffer.trim()) {
    const parsed = parseSseEvent(buffer);
    const finalResponse = parsed ? handleStreamEvent(parsed, streamNode) : null;
    if (finalResponse) return finalResponse;
  }

  throw new Error("流式连接已结束，但没有收到最终回答。");
}

async function sendChatMessageFallback(payload) {
  const response = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

function parseSseEvent(rawEvent) {
  const lines = rawEvent.split("\n");
  const dataLines = [];
  for (const line of lines) {
    if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
  }
  if (!dataLines.length) return null;
  return JSON.parse(dataLines.join("\n"));
}

function handleStreamEvent(event, streamNode) {
  if (event.event === "final") {
    return event.data?.response;
  }
  if (event.event === "error") {
    const detail = event.data?.detail ? `：${event.data.detail}` : "";
    throw new Error(`${event.message || "流式回答失败"}${detail}`);
  }
  appendStreamStep(streamNode, event);
  return null;
}

function appendAssistantComparison(left, right) {
  const node = document.createElement("article");
  node.className = "message assistant";
  node.innerHTML = `
    <div class="avatar">G</div>
    <div class="bubble">
      <div class="context-note">已选择 ${escapeHtml(left.name)} 和 ${escapeHtml(right.name)}。我先给你一个基础对比，接下来你可以像和导购聊天一样继续说预算、用途和顾虑。</div>
      ${renderComparison(left, right)}
    </div>
  `;
  chatStream.appendChild(node);
}

function renderComparison(left, right) {
  return `
    <header class="compare-head">
      <div>
        <h2>${escapeHtml(left.name)} vs ${escapeHtml(right.name)}</h2>
      </div>
    </header>
    <div class="compare-grid">
      ${renderCompareProduct(left)}
      ${renderCompareProduct(right)}
    </div>
    <div class="compare-summary">${escapeHtml(buildCompareSummary(left, right))}</div>
  `;
}

function renderCompareProduct(product) {
  const facts = buildProductPreview(product, 6);
  return `
    <article class="compare-product">
      <h3>${escapeHtml(product.name)}</h3>
      <div class="compare-meta">${escapeHtml(buildProductMetaLine(product) || "ZOL 参数库")}</div>
      <div class="compare-facts">
        ${facts.length ? facts.map((fact) => renderCompareFact(fact.label, fact.value)).join("") : renderCompareFact("数据", "已加载原始记录")}
      </div>
    </article>
  `;
}

function renderCompareFact(label, value) {
  return `<div class="compare-fact"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
}

function buildCompareSummary(left, right) {
  const leftMeta = buildProductMetaLine(left);
  const rightMeta = buildProductMetaLine(right);
  const preview = [leftMeta && `${left.name}：${leftMeta}`, rightMeta && `${right.name}：${rightMeta}`].filter(Boolean);
  return preview.length
    ? `${preview.join("；")}。最终推荐还要结合你的预算、用途和风险偏好。`
    : "最终推荐还要结合你的预算、用途和风险偏好。";
}

function appendUserMessage(message) {
  const node = document.createElement("article");
  node.className = "message user";
  node.innerHTML = `
    <div class="bubble">${escapeHtml(message)}</div>
    <div class="avatar">我</div>
  `;
  chatStream.appendChild(node);
  scrollToLatest();
}

function appendTypingMessage() {
  const node = document.createElement("article");
  node.className = "message assistant";
  node.innerHTML = `
    <div class="avatar">G</div>
    <div class="bubble">
      <div class="typing" aria-label="正在生成"><span></span><span></span><span></span></div>
    </div>
  `;
  chatStream.appendChild(node);
  scrollToLatest();
  return node;
}

function appendStreamingMessage() {
  const node = document.createElement("article");
  node.className = "message assistant";
  node.innerHTML = `
    <div class="avatar">G</div>
    <div class="bubble">
      <div class="stream-steps" aria-live="polite"></div>
      <div class="typing stream-typing" aria-label="正在生成"><span></span><span></span><span></span></div>
    </div>
  `;
  chatStream.appendChild(node);
  scrollToLatest();
  return node;
}

function appendStreamStep(node, event) {
  const list = node.querySelector(".stream-steps");
  if (!list) return;
  const item = document.createElement("div");
  item.className = `stream-step stream-step-${escapeCssClass(event.event)}`;
  item.textContent = event.message || "正在处理";
  list.appendChild(item);
  while (list.children.length > 6) list.firstElementChild?.remove();
  scrollToLatest();
}

function appendAssistantResponse(data) {
  const node = document.createElement("article");
  node.className = "message assistant";
  node.innerHTML = `
    <div class="avatar">G</div>
    <div class="bubble">${renderAssistantContent(data)}</div>
  `;
  chatStream.appendChild(node);
  scrollToLatest();
}

function renderFinalAssistantResponse(node, data) {
  const bubble = node.querySelector(".bubble");
  if (!bubble) return;
  bubble.innerHTML = renderAssistantContent(data);
  scrollToLatest();
}

function renderStreamError(node, error) {
  const bubble = node.querySelector(".bubble");
  if (!bubble) {
    appendErrorMessage(error);
    return;
  }
  bubble.innerHTML = renderErrorContent(error);
  scrollToLatest();
}

function appendErrorMessage(error) {
  const node = document.createElement("article");
  node.className = "message assistant";
  node.innerHTML = `
    <div class="avatar">G</div>
    <div class="bubble">${renderErrorContent(error)}</div>
  `;
  chatStream.appendChild(node);
  scrollToLatest();
}

function renderErrorContent(error) {
  return `
    <div class="error-box">
      <h2>需要调整一下</h2>
      <p>${escapeHtml(error.message || "未知错误")}</p>
    </div>
  `;
}

function renderAssistantContent(data) {
  return `<div class="plain-answer">${renderRichText(data.assistant_message)}</div>`;
}

function renderRichText(value) {
  // 先转义防 XSS，再把 **xxx** 还原成 <strong>，最后换行转 <br>。
  const escaped = escapeHtml(value ?? "");
  return escaped
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replaceAll("\n", "<br>");
}

function getSelectedProduct(input) {
  return pickers.get(input)?.selected || null;
}

function buildProductMetaLine(product, options = {}) {
  return buildProductPreview(product, 3, options)
    .map((entry) => `${entry.label}: ${entry.value}`)
    .join(" · ");
}

function getProductPrice(product) {
  const data = product.data || {};
  const raw = data.price_text || data.params?.电商报价 || "";
  const text = String(raw).trim();
  if (!text || /^[￥¥]?\s*0+(\.0+)?$/.test(text)) return "";
  return /^[￥¥]/.test(text) ? text : `￥${text}`;
}

function buildProductPreview(product, limit, options = {}) {
  const skipPrice = Boolean(options.skipPrice);
  const entries = flattenRecord(product.data || {})
    .filter((entry) => isUsefulPreview(entry, product.name))
    .filter((entry) => !skipPrice || !isPriceEntry(entry))
    .sort((a, b) => previewRank(b) - previewRank(a));
  return entries.slice(0, limit).map((entry) => ({
    label: prettifyLabel(entry.path),
    value: trimText(entry.value, 42),
  }));
}

function prettifyLabel(path) {
  // 扁平化后的 key 形如 "params.电商报价"，展示时去掉技术前缀，只保留末段字段名。
  return String(path).split(".").pop();
}

function isPriceEntry(entry) {
  return /电商报价|price/i.test(entry.path) || /[￥¥]\s*\d/.test(entry.value);
}

function flattenRecord(value, prefix = "") {
  if (Array.isArray(value)) {
    return value.flatMap((item, index) => flattenRecord(item, `${prefix}[${index}]`));
  }
  if (value && typeof value === "object") {
    return Object.entries(value).flatMap(([key, nested]) => flattenRecord(nested, prefix ? `${prefix}.${key}` : key));
  }
  if (value === null || value === undefined) return [];
  return [{ path: prefix, value: String(value).trim() }];
}

// 非规格字段：爬虫元数据 + 营销标签堆，展示时会显得突兀（如 price_text、使用场景），一律排除。
const PREVIEW_EXCLUDE_FIELDS = new Set([
  "zol_id",
  "title",
  "detail_url",
  "param_url",
  "price_text",
  "使用场景",
]);

function isUsefulPreview(entry, productName) {
  if (!entry.path || !entry.value) return false;
  if (entry.value === productName) return false;
  if (PREVIEW_EXCLUDE_FIELDS.has(prettifyLabel(entry.path))) return false;
  if (/^https?:\/\//i.test(entry.value)) return false;
  return entry.value.length <= 120;
}

function previewRank(entry) {
  const text = `${entry.path} ${entry.value}`;
  let score = 0;
  if (/[￥¥]\s*\d|\d+\s*元/.test(text)) score += 4;
  if (/20\d{2}\s*年|\d+\s*月/.test(text)) score += 3;
  if (/(gb|tb|mah|hz|克|g\b|cpu|ram|rom|ppi|英寸)/i.test(text)) score += 2;
  if (/id|url/i.test(entry.path)) score -= 3;
  return score;
}

function trimText(value, maxLength) {
  const text = String(value);
  return text.length > maxLength ? `${text.slice(0, maxLength - 1)}…` : text;
}

function autosizeTextarea() {
  userMessage.style.height = "auto";
  userMessage.style.height = `${Math.min(userMessage.scrollHeight, 160)}px`;
}

function setBusy(isBusy) {
  submitButton.disabled = isBusy;
}

function scrollToLatest() {
  requestAnimationFrame(() => window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" }));
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeCssClass(value) {
  return String(value).replace(/[^a-z0-9_-]/gi, "-");
}

autosizeTextarea();
loadProducts();
