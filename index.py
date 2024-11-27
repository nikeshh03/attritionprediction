import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))  # Load the pre-trained model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        # Retrieve form data
        Age = request.form.get("Age")
        BusinessTravel = request.form['BusinessTravel']
        DailyRate = request.form.get('DailyRate')
        Department = request.form['Department']
        DistanceFromHome = request.form.get("DistanceFromHome")
        Education = request.form.get("Education")
        EducationField = request.form['EducationField']
        EnvironmentSatisfaction = request.form.get("EnvironmentSatisfaction")
        Gender = request.form['Gender']
        HourlyRate = request.form.get("HourlyRate")
        JobInvolvement = request.form.get("JobInvolvement")
        JobLevel = request.form.get("JobLevel")
        JobRole = request.form['JobRole']
        JobSatisfaction = request.form.get("JobSatisfaction")
        MaritalStatus = request.form['MaritalStatus']
        MonthlyIncome = request.form.get("MonthlyIncome")
        NumCompaniesWorked = request.form.get("NumCompaniesWorked")
        OverTime = request.form['OverTime']
        PerformanceRating = request.form.get("PerformanceRating")
        RelationshipSatisfaction = request.form.get("RelationshipSatisfaction")
        StockOptionLevel = request.form.get("StockOptionLevel")
        TotalWorkingYears = request.form.get("TotalWorkingYears")
        TrainingTimesLastYear = request.form.get("TrainingTimesLastYear")
        WorkLifeBalance = request.form.get("WorkLifeBalance")
        YearsAtCompany = request.form.get("YearsAtCompany")
        YearsInCurrentRole = request.form.get("YearsInCurrentRole")
        YearsSinceLastPromotion = request.form.get("YearsSinceLastPromotion")
        YearsWithCurrManager = request.form.get("YearsWithCurrManager")

        # Create a dictionary to match the inputs to the expected format
        input_data = {
            'Age': int(Age),
            'BusinessTravel': str(BusinessTravel),
            'DailyRate': int(DailyRate),
            'Department': Department,
            'DistanceFromHome': int(DistanceFromHome),
            'Education': Education,
            'EducationField': str(EducationField),
            'EnvironmentSatisfaction': int(EnvironmentSatisfaction),
            'Gender': str(Gender),
            'HourlyRate': int(HourlyRate),
            'JobInvolvement': int(JobInvolvement),
            'JobLevel': int(JobLevel),
            'JobRole': JobRole,
            'JobSatisfaction': int(JobSatisfaction),
            'MaritalStatus': str(MaritalStatus),
            'MonthlyIncome': int(MonthlyIncome),
            'NumCompaniesWorked': int(NumCompaniesWorked),
            'OverTime': str(OverTime),
            'PerformanceRating': int(PerformanceRating),
            'RelationshipSatisfaction': int(RelationshipSatisfaction),
            'StockOptionLevel': StockOptionLevel,
            'TotalWorkingYears': int(TotalWorkingYears),
            'TrainingTimesLastYear': TrainingTimesLastYear,
            'WorkLifeBalance': int(WorkLifeBalance),
            'YearsAtCompany': int(YearsAtCompany),
            'YearsInCurrentRole': int(YearsInCurrentRole),
            'YearsSinceLastPromotion': int(YearsSinceLastPromotion),
            'YearsWithCurrManager': int(YearsWithCurrManager)
        }

        # Convert the dictionary into a pandas dataframe
        df = pd.DataFrame([input_data])

        # Feature Engineering
        # Calculate Total Satisfaction and convert it into a boolean
        df['Total_Satisfaction'] = (df['EnvironmentSatisfaction'] +
                                     df['JobInvolvement'] +
                                     df['JobSatisfaction'] +
                                     df['RelationshipSatisfaction'] +
                                     df['WorkLifeBalance']) / 5
        df['Total_Satisfaction_bool'] = df['Total_Satisfaction'].apply(lambda x: 1 if x >= 2.8 else 0)
        df.drop(['EnvironmentSatisfaction', 'JobInvolvement', 'JobSatisfaction', 'RelationshipSatisfaction', 'WorkLifeBalance', 'Total_Satisfaction'], axis=1, inplace=True)

        # Encode Age as boolean
        df['Age_bool'] = df['Age'].apply(lambda x: 1 if x < 35 else 0)
        df.drop('Age', axis=1, inplace=True)

        # Encode DailyRate as boolean
        df['DailyRate_bool'] = df['DailyRate'].apply(lambda x: 1 if x < 800 else 0)
        df.drop('DailyRate', axis=1, inplace=True)

        # Encode Department as boolean
        df['Department_bool'] = df['Department'].apply(lambda x: 1 if x == 'Research & Development' else 0)
        df.drop('Department', axis=1, inplace=True)

        # Encode DistanceFromHome as boolean
        df['DistanceFromHome_bool'] = df['DistanceFromHome'].apply(lambda x: 1 if x > 10 else 0)
        df.drop('DistanceFromHome', axis=1, inplace=True)

        # Encode JobRole as boolean
        df['JobRole_bool'] = df['JobRole'].apply(lambda x: 1 if x == 'Laboratory Technician' else 0)
        df.drop('JobRole', axis=1, inplace=True)

        # Encode HourlyRate as boolean
        df['HourlyRate_bool'] = df['HourlyRate'].apply(lambda x: 1 if x < 65 else 0)
        df.drop('HourlyRate', axis=1, inplace=True)

        # Encode MonthlyIncome as boolean
        df['MonthlyIncome_bool'] = df['MonthlyIncome'].apply(lambda x: 1 if x < 4000 else 0)
        df.drop('MonthlyIncome', axis=1, inplace=True)

        # Encode NumCompaniesWorked as boolean
        df['NumCompaniesWorked_bool'] = df['NumCompaniesWorked'].apply(lambda x: 1 if x > 3 else 0)
        df.drop('NumCompaniesWorked', axis=1, inplace=True)

        # Encode TotalWorkingYears as boolean
        df['TotalWorkingYears_bool'] = df['TotalWorkingYears'].apply(lambda x: 1 if x < 8 else 0)
        df.drop('TotalWorkingYears', axis=1, inplace=True)

        # Encode YearsAtCompany as boolean
        df['YearsAtCompany_bool'] = df['YearsAtCompany'].apply(lambda x: 1 if x < 3 else 0)
        df.drop('YearsAtCompany', axis=1, inplace=True)

        # Encode YearsInCurrentRole as boolean
        df['YearsInCurrentRole_bool'] = df['YearsInCurrentRole'].apply(lambda x: 1 if x < 3 else 0)
        df.drop('YearsInCurrentRole', axis=1, inplace=True)

        # Encode YearsSinceLastPromotion as boolean
        df['YearsSinceLastPromotion_bool'] = df['YearsSinceLastPromotion'].apply(lambda x: 1 if x < 1 else 0)
        df.drop('YearsSinceLastPromotion', axis=1, inplace=True)

        # Encode YearsWithCurrManager as boolean
        df['YearsWithCurrManager_bool'] = df['YearsWithCurrManager'].apply(lambda x: 1 if x < 1 else 0)
        df.drop('YearsWithCurrManager', axis=1, inplace=True)

        # One-hot Encoding for Categorical Variables
        # Business Travel
        if BusinessTravel == 'Rarely':
            df['BusinessTravel_Rarely'] = 1
            df['BusinessTravel_Frequently'] = 0
            df['BusinessTravel_No_Travel'] = 0
        elif BusinessTravel == 'Frequently':
            df['BusinessTravel_Rarely'] = 0
            df['BusinessTravel_Frequently'] = 1
            df['BusinessTravel_No_Travel'] = 0
        else:
            df['BusinessTravel_Rarely'] = 0
            df['BusinessTravel_Frequently'] = 0
            df['BusinessTravel_No_Travel'] = 1
        df.drop('BusinessTravel', axis=1, inplace=True)

        # Education (One-Hot Encoding)
        for i in range(1, 6):
            df[f'Education_{i}'] = 1 if Education == i else 0
        df.drop('Education', axis=1, inplace=True)

        # EducationField (One-Hot Encoding)
        education_fields = ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other']
        for field in education_fields:
            df[f'EducationField_{field.replace(" ", "_")}'] = 1 if EducationField == field else 0
        df.drop('EducationField', axis=1, inplace=True)

        # Gender (One-Hot Encoding)
        if Gender == 'Male':
            df['Gender_Male'] = 1
            df['Gender_Female'] = 0
        else:
            df['Gender_Male'] = 0
            df['Gender_Female'] = 1
        df.drop('Gender', axis=1, inplace=True)

        # Marital Status (One-Hot Encoding)
        for status in ['Married', 'Single', 'Divorced']:
            df[f'MaritalStatus_{status}'] = 1 if MaritalStatus == status else 0
        df.drop('MaritalStatus', axis=1, inplace=True)

        # Overtime (One-Hot Encoding)
        if OverTime == 'Yes':
            df['OverTime_Yes'] = 1
            df['OverTime_No'] = 0
        else:
            df['OverTime_Yes'] = 0
            df['OverTime_No'] = 1
        df.drop('OverTime', axis=1, inplace=True)

        # Stock Option Level (One-Hot Encoding)
        for i in range(4):
            df[f'StockOptionLevel_{i}'] = 1 if StockOptionLevel == i else 0
        df.drop('StockOptionLevel', axis=1, inplace=True)

        # Training Times Last Year (One-Hot Encoding)
        for i in range(7):
            df[f'TrainingTimesLastYear_{i}'] = 1 if TrainingTimesLastYear == i else 0
        df.drop('TrainingTimesLastYear', axis=1, inplace=True)

        # Make prediction
        prediction = model.predict(df)

        # Render prediction result
        if prediction == 0:
            return render_template('index.html', prediction_text='Employee Might Not Leave The Job')
        else:
            return render_template('index.html', prediction_text='Employee Might Leave The Job')

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

# # Run the application
# if __name__ == "__main__":
#     app.run(debug=True)

if __name__ == "__main__":
    app.run()

