from flask import Flask, request, jsonify
from groq import Groq

app = Flask(__name__)
client = Groq()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/ai", methods=["POST"])
def ai():
    data = request.get_json(force=True)
    text = data.get("text", "")

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": text}],
        temperature=1,
        max_completion_tokens=2048,
        top_p=1,
        stream=False,   # non-streaming, since server.js expects one JSON response
    )

    reply = completion.choices[0].message.content
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(__import__("os").environ.get("PORT", 5000)))
