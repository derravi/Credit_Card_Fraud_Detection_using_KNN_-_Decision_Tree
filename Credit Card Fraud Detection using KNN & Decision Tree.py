import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder  # UPDATED (merged imports)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import pickle

print("Lets Load the datasets........\n")
try:
    df = pd.read_csv("fraudTest.csv")
except FileNotFoundError:
    raise SystemExit("File not Exist: fraudTest.csv (place it next to this script or update path)")  # UPDATED
except Exception as e:
    raise SystemExit(f"An error occurred while reading CSV: {e}")  # UPDATED

print(f"The Total Number of the Rows is {df.shape[0]} and the Total Columns is {df.shape[1]}." )

print("The Data Types Of the all the Valuse:-\n")
df.dtypes

print("Lets Describe all the Int and Float Columns of the Datasets:-\n")
df.describe()

print("Lets see the Null value present in the Datasets:-\n")
df.isnull().sum()

print("Let see the data type of the all the Datasets.")
df.dtypes

# Drop unused columns
drop_cols = ["Unnamed: 0", "trans_date_trans_time", "cc_num", "first", "last", "street",
             "trans_num", "dob", "unix_time"]
existing_drop = [c for c in drop_cols if c in df.columns]  # UPDATED
df = df.drop(columns=existing_drop)

# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()  # UPDATED
print("Categorical columns found:", categorical_columns)

# Use separate LabelEncoder for each categorical column
encoders = {}  
for i in categorical_columns:
    le = LabelEncoder() 
    df[i] = df[i].astype(str) 
    df[i] = le.fit_transform(df[i]) 
    encoders[i] = le 

#Feture Selection
X = df.iloc[:,:-1]
y = df['is_fraud']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 

#Feture Scaling of the all data.
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)  
x_test_scaled = scaler.transform(x_test)

#Using the Knn Model.

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train_scaled, y_train)
y_pred_knn = knn.predict(x_test_scaled)
print("Classification Report of the KNN Model:\n", classification_report(y_test, y_pred_knn))
print("The Accuracy Score of the KNN Model is:\n", accuracy_score(y_test, y_pred_knn)*100)

print("The Accuracy Score of the KNN Model is:\n",accuracy_score(y_test,y_pred_knn)*100)

#Lets see the Confusion Matrix of this model Prediction

sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt="d", cmap='Blues')
plt.title("KNN Confusion Matrix")
plt.savefig("Diagram images/Predicted_Output_Confustion_Matrix.png", dpi=300, bbox_inches='tight')
plt.show()

dtree = DecisionTreeClassifier(max_depth=6, random_state=42)
dtree.fit(x_train, y_train)
y_tree_predict = dtree.predict(x_test)
print("Report of the Decision Tree Classifier:\n", classification_report(y_test, y_tree_predict))
print("Accuracy of the Decision Tree Classifier Model is:", accuracy_score(y_test, y_tree_predict)*100)


sns.heatmap(confusion_matrix(y_test, y_tree_predict), annot=True, fmt="d", cmap="Blues")
plt.title("Decision Tree Confusion Matrix")
plt.savefig("Diagram images/Confusion_Matrix_of_Decision_Tree_Classifier_Model.png", dpi=300, bbox_inches='tight')
plt.show()

fpr_knn, tpr_knn, _ = roc_curve(y_test, knn.predict_proba(x_test_scaled)[:,1])
fpr_dt, tpr_dt, _ = roc_curve(y_test, dtree.predict_proba(x_test)[:,1])

plt.figure(figsize=(6,6))
plt.plot(fpr_knn, tpr_knn, label="KNN (AUC = %0.2f)" % auc(fpr_knn, tpr_knn))
plt.plot(fpr_dt, tpr_dt, label="Decision Tree (AUC = %0.2f)" % auc(fpr_dt, tpr_dt))
plt.plot([0,1],[0,1],'--', color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.savefig("Diagram images/ROC_Curve_Comparison.png",dpi=500,bbox_inches='tight')
plt.show()

#User Input

def user_input_prediction(model,scaler):
    print("Enter the Transection Details:-")

    amt = float(input("Transaction Amount: "))
    category = input("Merchant Category (e.g., travel, health_fitness): ")
    gender = input("Gender (M/F): ")
    city_pop = int(input("City Population: "))
    lat = float(input("Latitude: "))
    long = float(input("Longitude: "))

    user_data = {
        "merchant":category,
        "category":category,
        "amt": amt,
        "gender":gender,
        "city_pop":city_pop,
        "lat":lat,
        "long":long,
        "merch_lat":lat + 0.01,
        "merch_long":long + 0.01
    }

    #Make the data frame of this user details. 
    input_data = pd.DataFrame([user_data])

    #Encode the Catagorical Column

    for i in categorical_columns:
        if i in input_data.columns:
            if i in encoders:
              input_data[i] = encoders[i].transform(input_data[i])
            else:
                 print(f"Warning: Encoder not found for {i}.")

    #Align With the Training Columns

    extra_cols = set(input_data.columns) - set(X.columns)
    input_data = input_data.drop(columns=extra_cols)

    missing_cols = set(X.columns) - set(input_data.columns)
    for j in missing_cols:
        input_data[j] = 0

    input_df = input_data[X.columns]

    #Scale Down the User input
    input_scaled = scaler.transform(input_df)

    #predict 

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        print("ALERT: Fraudulent Transaction Detected!")
    else:
        print("Transaction is Genuine.")
        
# Function Calling.

print("\nUser Prediction with Decision Tree.")
user_input_prediction(dtree, scaler)

os.makedirs("models", exist_ok=True)  
model_artifacts = {
    'encoders': encoders,          
    'feature_scaler': scaler,      
    'knn_model': knn,
    'dtree_model': dtree,
    'feature_columns': X.columns.tolist()  
}
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model_artifacts, f)  

print("Models saved successfully to models/model.pkl")