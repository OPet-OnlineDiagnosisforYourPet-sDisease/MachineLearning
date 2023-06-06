
import pandas as pd
import numpy as np
from keras.models import load_model

from flask import Flask, request, jsonify

app = Flask(__name__)

#Load the Model
model = load_model('dogModel.h5')

#Label
labels = ["Tick fever", "Distemper", "Parvovirus",
       "Hepatitis", "Tetanus", "Chronic kidney Disease", "Diabetes",
       "Gastrointestinal Disease", "Allergies", "Gingitivis", "Cancers",
       "Skin Rashes"]


@app.route("/predict", methods=["POST"])
def predict():
    json_ = request.json
    query_df = pd.DataFrame(json_)

    #Jika input < 3 maka menampilkan error
    #if query_df.shape[1] < 3:
        #return jsonify({"message": "Maaf, setidaknya Anda perlu melakukan input 3 gejala."})

    # Check if input has at least 3 symptoms
    #if len(json_) < 3:
        #return jsonify({"error": "Maaf, setidaknya Anda harus menginput 3 gejala."})


    #Merubah input dalam bentuk array
    df = [np.array(query_df)]

    #Prediksi
    prediction = model.predict(df)

    # Menentukan label berdasarkan nilai prediksi tertinggi
    max_index = np.argmax(prediction)
    predicted_label = labels[max_index]

    #Menampilkan prediksi menjadi list
    prediction_list = prediction.tolist()

    return jsonify({"Prediction": (predicted_label)})
    

if __name__ == '__main__':
    app.run()


