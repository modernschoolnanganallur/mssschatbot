// ========= MAIN.JS (Netlify -> Cloud Run) =========

// 1) Use Netlify proxy. Your _redirects already points /api/* to Cloud Run.
const API = "/api";

// --- Small status banner so you can see connectivity at a glance
(function ensureStatusBanner() {
  if (!document.getElementById("status-banner")) {
    const el = document.createElement("div");
    el.id = "status-banner";
    el.style.cssText =
      "position:fixed;bottom:8px;left:8px;padding:6px 10px;font:12px/1.4 system-ui;background:#111;color:#fff;border-radius:6px;z-index:99999;opacity:.9";
    el.textContent = "main.js loaded…";
    document.addEventListener("DOMContentLoaded", () =>
      document.body.appendChild(el)
    );
  }
})();
function showStatus(msg) {
  console.log(msg);
  const b = document.getElementById("status-banner");
  if (b) b.textContent = msg;
}

// --- Health check
async function ping() {
  try {
    const res = await fetch(`${API}/health`, { cache: "no-store" });
    if (!res.ok) throw new Error(`Health ${res.status}`);
    const data = await res.json();
    console.log("✅ Backend Health:", data);
    showStatus("✅ Backend OK");
  } catch (err) {
    console.error("❌ Backend not reachable:", err);
    showStatus("⚠️ Server not reachable.");
  }
}

// --- Ask endpoint
async function ask(question) {
  try {
    const res = await fetch(`${API}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question })
    });
    if (!res.ok) throw new Error(`Ask failed: ${res.status}`);
    const data = await res.json();
    return data.answer;
  } catch (err) {
    console.error(err);
    showStatus("⚠️ Assistant server error");
    return "Sorry — I couldn’t reach the assistant. Please try again.";
  }
}

// --- UI wiring
document.addEventListener("DOMContentLoaded", () => {
  ping(); // test immediately

  const form = document.getElementById("chat-form");
  const input = document.getElementById("user-input");
  const chatBox = document.getElementById("chat-box");

  if (!form || !input || !chatBox) {
    console.warn("Chat elements not found in DOM.");
    showStatus("⚠️ Chat elements not found");
    return;
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = input.value.trim();
    if (!question) return;

    chatBox.innerHTML += `<div class="msg user">You: ${question}</div>`;
    input.value = "";
    const reply = await ask(question);
    chatBox.innerHTML += `<div class="msg bot">Brightly: ${reply}</div>`;
    chatBox.scrollTop = chatBox.scrollHeight;
  });
});
