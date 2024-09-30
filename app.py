from flask import Flask, jsonify, request
import requests

from dataset import transformation, train_dataset
from model import model
from results_eval import test_image, predict, preprocess_image
from training_loop import device

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def get_image_prediction():
    if request.method == 'POST':
        if "file" not in request.files:
            return jsonify({"error": "No file found"}), 400
        file = request.files["file"]
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        image, tensor = preprocess_image(image_path=file, transform=transformation)
        probabilities = predict(model=model, image_tensor=tensor, device=device)
        class_names = train_dataset.classes
        predicted_class = class_names[probabilities.argmax()]
        probability = probabilities[0].item()
        return jsonify({
            "image": file.filename,
            "predicted_class": predicted_class,
            "probability": probability
        })


#  TODO response = requests.post(url="http://127.0.0.1:5000/predict", files={'file': open(test_image, 'rb')})


if __name__ == '__main__':
    app.run(debug=True)
