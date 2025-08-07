import os
import gradio as gr
import openai
import google.generativeai as genai

# --- Load API keys from environment ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Configure APIs ---
openai.api_key = OPENROUTER_API_KEY
openai.api_base = "https://openrouter.ai/api/v1"
genai.configure(api_key=GEMINI_API_KEY)

# --- Model choices ---
model_choices = [
    ("gpt-3.5-turbo", "🔷 OpenAI GPT-3.5 Turbo via OpenRouter"),
    ("gemini-1.5-pro", "🔶 Gemini 1.5 Pro")
]

# --- Unified chat function ---
def unified_chat(message, history, model_selection):
    if model_selection == "gpt-3.5-turbo":
        try:
            messages = [{"role": "user", "content": message}]
            for user_msg, assistant_msg in history:
                messages.insert(-1, {"role": "assistant", "content": assistant_msg})
                messages.insert(-1, {"role": "user", "content": user_msg})

            response = openai.ChatCompletion.create(
                model="openai/gpt-3.5-turbo",
                messages=messages
            )
            reply = response['choices'][0]['message']['content']
            return reply
        except Exception as e:
            return f"❌ OpenRouter GPT Error: {str(e)}"

    elif model_selection == "gemini-1.5-pro":
        try:
            model = genai.GenerativeModel("gemini-1.5-pro")
            convo = model.start_chat()
            for user_msg, assistant_msg in history:
                convo.send_message(user_msg)
                convo.send_message(assistant_msg)

            response = convo.send_message(message)
            return response.text
        except Exception as e:
            return f"❌ Gemini Error: {str(e)}"
    else:
        return "❌ Error: Invalid model selected."

# --- Gradio UI ---
with gr.Blocks(css="""
body {
    margin: 0;
    padding: 0;
    background: linear-gradient(to right, #f0f4ff, #fafcff);
    font-family: 'Segoe UI', sans-serif;
}
#chat-heading {
    text-align: center;
    font-size: 2.5em;
    font-weight: bold;
    color: #333;
    background: linear-gradient(90deg, #6366f1, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 1.5rem 0;
}
.gr-button {
    border-radius: 10px !important;
    background: linear-gradient(to right, #6366f1, #60a5fa) !important;
    color: white !important;
}
.gr-textbox textarea {
    background-color: white !important;
    border-radius: 10px !important;
    border: 1px solid #ccc !important;
}
.gr-dropdown {
    border-radius: 10px !important;
}
.gr-chatbot {
    background-color: #ffffffdd !important;
    border-radius: 15px !important;
    padding: 1rem !important;
    border: 1px solid #ddd !important;
    box-shadow: 0 8px 16px rgba(0,0,0,0.08) !important;
    width: 100% !important;
    max-height: 600px;
    overflow-y: auto;
}
""") as demo:

    gr.Markdown("<div id='chat-heading'>🤖 AI ChatBot UI</div>")

    with gr.Row():
        model_selector = gr.Dropdown(
            choices=[choice[0] for choice in model_choices],
            label="🧠 Select your AI model",
            value="gpt-3.5-turbo",
            scale=1
        )

    chatbot = gr.Chatbot(label="💬 Chat Window", height=500, show_label=True)

    msg = gr.Textbox(
        label="💬 Type your message and press Enter",
        placeholder="Start chatting with GPT or Gemini...",
        lines=1  # Single-line input
    )

    def respond(message, history, model_selection):
        reply = unified_chat(message, history, model_selection)
        history.append((message, reply))
        return history, ""

    msg.submit(respond, [msg, chatbot, model_selector], [chatbot, msg])

demo.launch()
