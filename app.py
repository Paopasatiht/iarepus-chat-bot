from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chat import _get_response
from utils.dialogue_manager import DialogueManager
from sentence_transformers import SentenceTransformer
from utils.yamlparser import YamlParser

# Declare a Flask app :
app = Flask(__name__)
CORS(app)
config_file = "/Projects/configs/config.yaml"
cfg = YamlParser(config_file)

model = SentenceTransformer(cfg["MODEL"]["answer_model"])

# Main function here :
@app.get("/")
def index_get():
    return render_template("base.html")


@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    msg_manager = DialogueManager(model)
    response = _get_response(text, msg_manager)
    message =  {"answer" : response}
    return jsonify(message)

# Runing the app :
if __name__ == "__main__":
    app.run(debug=True)