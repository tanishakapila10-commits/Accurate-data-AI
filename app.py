import gradio as gr
import pandas as pd
import joblib

model = joblib.load("student_performance_model.pkl")

def predict_performance(
    Age,
    Gender,
    Attendance_Rate,
    Prior_GPA,
    LMS_Logins,
    Study_Hours_Per_Week,
    Assignment_Submission_Rate,
    Engagement_Score
):

    data = pd.DataFrame([{
        "Age": Age,
        "Gender": Gender,
        "Attendance_Rate": Attendance_Rate,
        "Prior_GPA": Prior_GPA,
        "LMS_Logins": LMS_Logins,
        "Study_Hours_Per_Week": Study_Hours_Per_Week,
        "Assignment_Submission_Rate": Assignment_Submission_Rate,
        "Engagement_Score": Engagement_Score
    }])

    prediction = model.predict(data)[0]

    return prediction


app = gr.Interface(
    fn=predict_performance,

    inputs=[
        gr.Number(label="Age"),
        gr.Dropdown(["Female", "Male"], label="Gender"),
        gr.Number(label="Attendance Rate"),
        gr.Number(label="Prior GPA"),
        gr.Number(label="LMS Logins"),
        gr.Number(label="Study Hours Per Week"),
        gr.Number(label="Assignment Submission Rate"),
        gr.Number(label="Engagement Score")
    ],

    outputs=gr.Textbox(label="Predicted Academic Performance"),

    title="Student Academic Performance Prediction",

    description="Enter student details to predict academic performance."
)

app.launch()
