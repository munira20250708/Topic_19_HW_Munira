import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)
loaded_model = joblib.load('model_classifier_rf.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = ['Age', 'RestingBP', 'FastingBS', 'Oldpeak', 'Sex_F','Sex_M', 
                           'ChestPainType_ASY', 'ChestPainType_ATA', 'ChestPainType_NAP',
                           'ChestPainType_TA', 'RestingECG_LVH', 'RestingECG_Normal',
                           'RestingECG_ST', 'ST_Slope_Down', 'ST_Slope_Flat', 'ST_Slope_Up']
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = loaded_model.predict(df)
        
    if output == 1:
        res_val = "** heartdisease **"
    else:
        res_val = "no heartdisease"
        

    return render_template('index.html', prediction_text='Patient has {}'.format(res_val))

if __name__ == "__main__":
    app.run()