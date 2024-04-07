from flask import Flask, request, jsonify
from flask_cors import CORS 
import pickle

app = Flask(__name__)
CORS(app)  


with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


@app.route('/predict', methods=['POST'])
def predict():
    try:
       
        data = request.json

       
        features = [data['temperature'], data['humidity'], data['wind_speed']]

        predictions = model.predict([features])

        predicted_shirt = predictions[:, 0][0]
        predicted_jeans = predictions[:, 1][0]

        response = {
            'shirt': predicted_shirt,
            'jeans': predicted_jeans
        }

        return jsonify(response), 200

    except Exception as e:

        error_message = {'error': str(e)}
        return jsonify(error_message), 500

if __name__ == '__main__':
    app.run(debug=True)
