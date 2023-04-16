import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
# Load the model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))


@app.route('/')
def home():
    """
    render_template will look for templates folder
    """
    return render_template('home.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    """
    Input in json format and captured inside data key and store in data
    new_data is transformed data
    """
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])


if __name__ == "__main__":
    app.run(debug=True)
