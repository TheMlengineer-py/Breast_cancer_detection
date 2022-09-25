import pickle
from flask import Flask, render_template, request, app, jsonify, url_for
import numpy as np
import pandas as pd 


app = Flask(__name__,template_folder='Templates')

model   = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict_api',methods=['POST'])
def predict_api():

    data=request.json['data']
    print(data)
    new_data=[list(data.values())]
    output=round(model.predict(new_data)[0])
    if output == 1:
        # val_ = 'Yes'
        output = 'Malignant_tumor'
    else:
        # val_ = 'no'
        output = 'Benign_tumor'
    return jsonify(output)

@app.route('/predict',methods=['POST'])
def predict():

    data=[float(x) for x in request.form.values()]
    final_features = [np.array(data)]
    print(data)
    
    output=model.predict(final_features)[0]
    # print(output[0])
    if round(output) == 1:
        val_ = 'Yes'
        output = 'Malignant_tumor'
    else:
        val_ = 'no'
        output = 'Benign_tumor'

    return render_template('index.html', prediction_text="Model Predict : {0}, this is {1}.".format(val_, output))


if __name__ == "__main__":
    app.run(debug=True)
