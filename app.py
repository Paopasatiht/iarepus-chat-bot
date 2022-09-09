from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chat import _get_response
from utils.dialogue_manager import DialogueManager

app = Flask(__name__)
CORS(app)


@app.get("/")
def index_get():
    return render_template("base.html")


@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    # TODO Check if the text is valid
    msg_manager = DialogueManager()
    response = _get_response(text, msg_manager)
    message =  {"answer" : response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)