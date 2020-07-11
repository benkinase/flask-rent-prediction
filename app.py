import model
from flask import Flask, request, jsonify, render_template, url_for
import numpy as np
import pickle

# Web app for the rent prediction
app = Flask(__name__)


# load the model from disk
filename = 'model.sav'
loaded_model = pickle.load(open(filename, 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    int_features = list(map(int, features))

    # print(int_features)
    numpy_features = [np.array(int_features)]
    prediction = loaded_model.predict(numpy_features)
    # print(prediction)
    print(loaded_model.coef_)

    return render_template("index.html",  prediction_value="Predicted rent: $%.2f" % prediction)


if __name__ == "__main__":
    app.run(debug=True)
