from fastapi import FastAPI
from Schema.pydentic_model import UserInput
from fastapi.responses import JSONResponse
import pandas as pd
import pickle

app = FastAPI(title="Credit Card Fraud Detection using KNN & Decision Tree")

#Load the Pickle Model and import all the ML Models.
with open('models/model.pkl','rb') as f:
    model = pickle.load(f)

encoders_ = model['encoders']     
le = model['labelencoder']
scaler = model['scaler']
knn = model['knn_model']
dtree = model['dtree_model']

@app.get("/")
def demo():
    return {"message":"Credit Card Fraud Detection using KNN & Decision Tree"}

@app.post("/predict")
def predict(predict:UserInput):
    
    df = pd.DataFrame([{
        'merchant': predict.merchant, #float
        'category':predict.category, #string
        'amt':predict.amt,  #float
        'gender':predict.gender, #string
        'city_pop':predict.city_pop, #int
        'lat':predict.lat, #int
        'long':predict.long, #float
        'merch_lat':predict.merch_lat, #float
        'merch_long':predict.merch_long #float
    }])

    #Using the LabelEncoder we are Encode the 'category', 'merchant' and 'gender'
    for j in ['category', 'gender', 'merchant']:
        if j in encoders_:
            classes = encoders_[j].classes_
            df[j] = df[j].apply(
                lambda x: encoders_[j].transform([x])[0] if x in classes else -1
            )
            unseen = [x for x in df[j] if x == -1]
            if unseen:
                print(f"Warning: Unseen label(s) in '{j}', replaced with -1.")
        else:
            print(f"Encoder not found for column '{j}'.")

    feature_columns = model['feature_columns']

    # Add missing columns with 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    #Drope Unwanted Columns
    df = df[feature_columns]

    #Lets Scale down the all user data
    new_df = scaler.transform(df)

    print("lets Predict the Fraude Detection using the 2 Models........\n")

    #Using the knn Model

    knn_prediction = knn.predict(new_df)
    dtree_predictino = dtree.predict(new_df)

    #Lets Use the List Comprehention
    temp1 = "ALERT: Fraudulent Transaction Detected!" if knn_prediction[0] == 1 else "Transaction is Genuine."

    temp2 = "ALERT: Fraudulent Transaction Detected!" if dtree_predictino[0] == 1 else "Transaction is Genuine."

    return JSONResponse(status_code=200,content={
        "The Prediction of the KNN Model": temp1,
        "The Prediction of the Decision Tree Classifier":temp2
    })