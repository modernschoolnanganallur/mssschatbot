// js/config.js
const API = "https://modernschool-chatbot-587062542199.asia-south1.run.app"; // Cloud Run URL

// Function to send a chat message to backend
async function sendMessage(userMessage, chatHistory = []) {
  try {
    const response = await fetch(`${API}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: userMessage,
        history: chatHistory,
      }),
    });

    if (!response.ok) {
      throw new Error(`Server returned ${response.status}`);
    }

    const data = await response.json();
    return data.reply || "No response received.";
  } catch (error) {
    console.error("Error contacting chatbot API:", error);
    return "There was an issue connecting to the chatbot.";
  }
}

// Export function for use elsewhere
export { sendMessage };
