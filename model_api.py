import joblib
import warnings
import pandas as pd
from flask import Flask, request
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore")
app = Flask(__name__)

df = pd.read_csv("labeldata.csv")
label_encoders = {}
for col in ['Experience', 'Education', 'Skills', 'JobTitle']:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

@app.get("/api/model")
def model_api():
    experience = request.form['experience'].lower()
    education = request.form['education'].lower().removesuffix("'s")
    yearofexperience = request.form['yearofexperience'].lower()
    skills = request.form['skills'].lower()
    
    new_data = {"Experience": experience, "Education": education, "YearsExperience": int(yearofexperience), "Skills": skills}

    for col in ['Experience', 'Education', 'Skills']:
        try:
            new_data[col] = label_encoders[col].transform([new_data[col]])[0]
        except ValueError as e:
            new_data[col] = -1
    
    new_data_array = [[new_data['Experience'], new_data['Education'], new_data['YearsExperience'], new_data['Skills']]]
    loaded_model = joblib.load('model.pkl')
    predicted_job_title = label_encoders['JobTitle'].inverse_transform(loaded_model.predict(new_data_array))[0]
    job_title = {"job_title": predicted_job_title.title()}
    return job_title

if __name__ == "__main__":
    app.run(debug=False)