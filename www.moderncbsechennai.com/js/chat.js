// js/chat.js

window.addEventListener("DOMContentLoaded", () => {
  // ---- Config / Base URL --------------------------------------------------
  // Prefer API from js/config.js; otherwise fall back to Netlify proxy (/api)
const API = "https://chatbot-api-587062542199.asia-south1.run.app";
console.log("[chat.js] Using Cloud Run API:", API);


  if (typeof API === "undefined") {
    console.warn(
      "[chat.js] API not found; using fallback:", API
    );
  } else {
    console.log("[chat.js] Using API:", API);
  }

  // ---- DOM refs -----------------------------------------------------------
  const chatBtn    = document.getElementById("chat-btn");
  const chatWindow = document.getElementById("chat-window");
  const sendBtn    = document.getElementById("send-btn");
  const userInput  = document.getElementById("user-input");
  const chatBody   = document.getElementById("chat-body");
  const header     = document.getElementById("chat-header");

  if (!chatBtn || !chatWindow || !sendBtn || !userInput || !chatBody || !header) {
    console.error("[chat.js] Missing one or more required DOM elements.");
    return;
  }

  // ---- State --------------------------------------------------------------
  let waiting = false;
  let chatOpen = false;
  const chatHistory = [{ type: "bot", text: "Hello! Ask me about school." }];

  // ---- UI helpers ---------------------------------------------------------
  function renderMessages() {
    chatBody.innerHTML = "";
    chatHistory.forEach((msg) => {
      const bubble = document.createElement("div");
      bubble.className = `message ${msg.type}`;
      if (msg.type === "bot") {
        const icon = document.createElement("span");
        icon.className = "icon";
        icon.textContent = "💡";
        bubble.appendChild(icon);
      }
      bubble.appendChild(document.createTextNode(msg.text));
      chatBody.appendChild(bubble);
    });
    chatBody.scrollTop = chatBody.scrollHeight;
  }

  function autoResizeTextarea() {
    userInput.style.height = "auto";
    userInput.style.height = Math.min(userInput.scrollHeight, 120) + "px";
  }

  // ---- Draggable window ---------------------------------------------------
  let isDragging = false,
    offsetX = 0,
    offsetY = 0;

  header.addEventListener("mousedown", (e) => {
    isDragging = true;
    offsetX = e.clientX - chatWindow.offsetLeft;
    offsetY = e.clientY - chatWindow.offsetTop;
    header.style.cursor = "grabbing";
  });
  document.addEventListener("mousemove", (e) => {
    if (isDragging) {
      chatWindow.style.left = e.clientX - offsetX + "px";
      chatWindow.style.top = e.clientY - offsetY + "px";
    }
  });
  document.addEventListener("mouseup", () => {
    isDragging = false;
    header.style.cursor = "grab";
  });

  // ---- Toggle open/close --------------------------------------------------
  chatBtn.addEventListener("click", () => {
    chatOpen = !chatOpen;
    chatWindow.style.display = chatOpen ? "flex" : "none";
    if (chatOpen) {
      autoResizeTextarea();
      userInput.focus();
    }
  });

  // ---- API helper ---------------------------------------------------------
  async function postJSON(path, bodyObj) {
    const url = `${API}${path}`;
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(bodyObj),
      // credentials: "include", // not needed unless you’re using cookies
    });
    if (!res.ok) {
      const txt = await res.text().catch(() => "");
      throw new Error(`${url} -> ${res.status} ${res.statusText} ${txt}`);
    }
    return res.json();
  }

  // ---- Send flow ----------------------------------------------------------
  async function sendMessage() {
    if (waiting) return;
    const text = (userInput.value || "").trim();
    if (!text) return;

    chatHistory.push({ type: "user", text });
    renderMessages();

    userInput.value = "";
    autoResizeTextarea();

    chatHistory.push({ type: "bot", text: "💭 Thinking..." });
    renderMessages();
    waiting = true;
    sendBtn.disabled = true;

    try {
      const data = await postJSON("/ask", { question: text });
      chatHistory.pop(); // remove "Thinking…"
      chatHistory.push({
        type: "bot",
        text: data?.answer || "I couldn't find an answer.",
      });
      renderMessages();
    } catch (e) {
      console.error(e);
      chatHistory.pop();
      chatHistory.push({
        type: "bot",
        text:
          "⚠️ I’m having trouble reaching the server right now. Please try again.",
      });
      renderMessages();
    } finally {
      waiting = false;
      sendBtn.disabled = false;
    }
  }

  // ---- Wire events --------------------------------------------------------
  sendBtn.addEventListener("click", sendMessage);
  userInput.addEventListener("input", autoResizeTextarea);
  userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // ---- Boot ---------------------------------------------------------------
  renderMessages();
  autoResizeTextarea();

  // Quick health ping
  fetch(`${API}/health`)
    .then((r) => r.json())
    .then((d) => console.log("[chat.js] Backend health:", d))
    .catch(() => console.warn("[chat.js] Backend not reachable"));
});
