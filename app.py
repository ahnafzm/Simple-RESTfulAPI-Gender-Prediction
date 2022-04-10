from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def Gender_prediction():
    if request.method == 'GET':
        return render_template("Gender-prediction.html")
    elif request.method == 'POST':
        print(dict(request.form))
        gender_features = dict(request.form).values()
        gender_features = np.array([float(x) for x in gender_features])
        model, std_scaler = joblib.load("model-development/Gender-classification-using-logistic-regression.pkl")
        gender_features = std_scaler.transform([gender_features])
        print(gender_features)
        result = model.predict(gender_features)
        gender = {
            '0': 'Female',
            '1': 'Male'
        }
        result = gender[str(result[0])]
        return render_template('Gender-prediction.html', result=result)
    else:
        return "Unsupported Request Method"


if __name__ == '__main__':
    app.run(port=5000, debug=True)