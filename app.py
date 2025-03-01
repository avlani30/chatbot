from flask import Flask, request, jsonify, render_template
import chatbot  # Import the chatbot logic from chatbot.py

app = Flask(__name__)

# Route to serve the main UI page.
@app.route("/")
def index():
    return render_template("index.html")

# API endpoint to receive a question and return an answer.
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"answer": "No question provided."})
    answer = chatbot.chatbot_response(question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
