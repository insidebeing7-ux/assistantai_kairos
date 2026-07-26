from flask import Flask, request, jsonify
from groq import Groq

app = Flask(__name__)
client = Groq()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

import json
import re

SYSTEM_PROMPT = """You are a personal assistant embedded in a messaging app.
You can only answer questions using the context given to you in the prompt.

If — and only if — the user is clearly asking you to COMPOSE/SEND a message
to one of their contacts (e.g. "tell Sam I'll be late", "send John a message
saying hi"), respond with ONLY a JSON object, no other text, in this exact
shape:
{"action":"send_message","target_username":"<username from context>","draft":"<the message text you composed>","reply":"<short confirmation text to show the user>"}

The target_username MUST be a username that literally appears in the
"Recent conversations" or "Recent messages with X" context you were given.
Never invent a username. If you can't confidently match one, do NOT use the
send_message action — just answer normally instead.

For every other question, just answer normally as plain text (no JSON)."""

@app.route("/ai", methods=["POST"])
def ai():
    data = request.get_json(force=True)
    text = data.get("text", "")
    mode = data.get("mode", "")

    messages = [{"role": "user", "content": text}]
    if mode == "personal_assistant":
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ]

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=1,
        max_completion_tokens=2048,
        top_p=1,
        stream=False,
    )
    reply = completion.choices[0].message.content

    # Try to detect a send_message action the model returned as JSON.
    action = None
    target_username = None
    draft = None
    if mode == "personal_assistant":
        candidate = reply.strip()
        match = re.search(r'\{[\s\S]*"action"\s*:\s*"send_message"[\s\S]*\}', candidate)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if parsed.get("action") == "send_message" and parsed.get("target_username") and parsed.get("draft"):
                    action = "send_message"
                    target_username = parsed["target_username"]
                    draft = parsed["draft"]
                    reply = parsed.get("reply") or f"Ready to send to {target_username}: \"{draft}\""
            except Exception:
                pass

    resp = {"reply": reply}
    if action == "send_message":
        resp["action"] = "send_message"
        resp["target_username"] = target_username
        resp["draft"] = draft
    return jsonify(resp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(__import__("os").environ.get("PORT", 5000)))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(__import__("os").environ.get("PORT", 5000)))
