import pandas as pd
import numpy as np
import pickle
from flask import Flask, request
from baja_competitions import BajaCompetitions

# Load model
model = pickle.load(open('../persistence/model.pkl', 'rb'))

# Instanciate Flask
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    test_json = request.get_json()

    # Collect data
    if test_json:
        if isinstance(test_json, dict):
            df_raw = pd.DataFrame(test_json, index=[0])
        else:
            df_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

    # Instatiate data preparation
    pipeline = BajaCompetitions()

    # Data preparation
    df_mod = pipeline.data_preparation(df=df_raw)

    # Prediction
    pred = model.predict(df_mod)

    df_raw['prediction'] = np.ceil(pred)

    return df_raw.to_json(orient='records')


if __name__ == '__main__':
    # Start Flask
    app.run(host='127.0.0.1',
            port='5000')
