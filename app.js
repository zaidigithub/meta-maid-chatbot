const chatForm = document.getElementById("chat-form");
const messageInput = document.getElementById("message");
const chatMessages = document.getElementById("chat-messages");

const history = [
  {
    role: "bot",
    text: "Hello! I am MetaMaid. Tell me your symptoms like fever, cough, headache, nausea, or sore throat.",
  },
];

function addMessage(role, text) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.textContent = text;

  article.appendChild(bubble);
  chatMessages.appendChild(article);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function formatReply(data) {
  const lines = [data.message];

  if (data.top_disease && data.top_disease !== "Greeting" && data.top_disease !== "Help") {
    lines.push(`Predicted disease: ${data.top_disease}`);
  }
  if (Array.isArray(data.matched_symptoms) && data.matched_symptoms.length > 0) {
    lines.push(`Matched symptoms: ${data.matched_symptoms.join(", ")}`);
  }

  if (Array.isArray(data.precautions) && data.precautions.length > 0) {
  lines.push(`Precautions: ${data.precautions.join(" | ")}`);
  }

if (data.advice) {
  lines.push(`Advice: ${data.advice}`);
  }

if (data.recommended_doctor) {
  lines.push(`Recommended doctor: ${data.recommended_doctor}`);
  }


  return lines.join("\n");
}

function autoResize() {
  messageInput.style.height = "auto";
  messageInput.style.height = `${messageInput.scrollHeight}px`;
}

messageInput.addEventListener("input", autoResize);

messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const message = messageInput.value.trim();
  if (!message) {
    return;
  }

  addMessage("user", message);
  history.push({ role: "user", text: message });
  messageInput.value = "";
  autoResize();

  const submitButton = chatForm.querySelector("button");
  submitButton.disabled = true;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message,
        history,
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Something went wrong.");
    }

    const botReply = formatReply(data);
    addMessage("bot", botReply);
    history.push({ role: "bot", text: botReply });
  } catch (error) {
    const fallback = error instanceof Error ? error.message : "Unable to reach the chatbot.";
    addMessage("bot", `Error: ${fallback}`);
  } finally {
    submitButton.disabled = false;
    messageInput.focus();
  }
});
