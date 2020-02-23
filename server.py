from flask import Flask, jsonify
import pandas as pd
import healthcareai

app = Flask(__name__)


#Yahan pr menay apna already saved/trained model load krlia kisi bhi variable main
trained_model = healthcareai.load_saved_model('2020-02-22T19-17-50_classification_RandomForestClassifier.pkl')

#Yeh mera predictions krne k liye data, json format main bhi ho sakta haii; menay yahan pr simple dataset hi utha lia
prediction_dataframe = healthcareai.load_diabetes()

@app.route("/")
def hello(): 
    return "Hello Friend!"

@app.route("/predict")
def predict():
    #Yahan pr jo data aya tha usko menay model pr laga dia
    predictions = trained_model.make_predictions(prediction_dataframe)
    #Yahan pr menay result ko JSON main convert kiya aur jahan sy request ayi thi wapis return krdia
    print(predictions)
    return jsonify({'predictions': list(predictions)})

if __name__ == '__main__':
    app.run(debug=True)