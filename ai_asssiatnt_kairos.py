from flask import Flask, request, jsonify
from groq import Groq

app = Flask(__name__)
client = Groq()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

import json
import re

SYSTEM_PROMPT = """You are Kairos, a personal assistant embedded in a messaging app.
You can only answer using the context given to you in the prompt, plus the
short recent conversation history (up to your last 5 turns with this user)
that is included below the context — use that history to resolve references
like "him", "her", "them", "that person" to whoever was actually named
earlier in the conversation.

SENDING MESSAGES — READ CAREFULLY:
You can only send a message if the user has given you the contact's EXACT
username as it appears in "Recent conversations" or "Recent messages with X"
in the context. Nicknames, first names, or vague references ("my friend",
"him", "that guy") are NOT enough on their own — but if the recent history
below already resolved that reference to a real username, you may use it.

If the user asks you to send/tell/message someone AND you can confidently
resolve their exact username from the context or recent history, respond
with ONLY a JSON object, no other text, in this exact shape:
{"action":"send_message","target_username":"<exact username from context>","draft":"<the message text you composed>","reply":"<short confirmation text to show the user>"}

If the user asks you to send a message but you are NOT sure which contact
they mean (name doesn't clearly match anyone in context, or no name was
given at all), do NOT guess and do NOT use the send_message action. Instead
reply normally, in plain text, asking them to confirm the exact username —
for example: "Who would you like me to send that to? Just give me their
exact username and what you'd like it to say." If it would help, you can
also mention 1-2 usernames from their recent conversations as likely
candidates, but only ones that actually appear in the context.

Never invent a username that doesn't appear in the context.
For every other question, just answer normally as plain text (no JSON)."""

@app.route("/ai", methods=["POST"])
def ai():
    data = request.get_json(force=True)
    text = data.get("text", "")
    mode = data.get("mode", "")
    # NEW — short rolling history from the client, capped defensively here too
    # (client already caps at 5, but never trust the caller alone).
    history = data.get("history", [])
    if not isinstance(history, list):
        history = []
    history = history[-5:]

    messages = [{"role": "user", "content": text}]
    if mode == "personal_assistant":
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in history:
            role = turn.get("role")
            content = turn.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                messages.append({"role": role, "content": content[:2000]})
        messages.append({"role": "user", "content": text})
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
