# Stress-Prediction-Model
Machine Learning model predicting stress based on lifestyle habit data.

Stress Level Prediction Using Daily Lifestyle Data

This project predicts daily stress levels based on lifestyle patterns such as sleep, work duration, physical activity, caffeine intake, screen time & more.
The dataset is self-collected, making this project unique, original, and closer to a real personal-wellness use case than typical Kaggle-trained models.


Project Overview

Built a machine learning model to predict stress level (0–10)

Dataset contains 11 lifestyle + contextual features

Includes visual analysis, model comparison, and feature insights

Shows which habits contribute most to daily stress levels

This project demonstrates data understanding, modeling thinking and health-behavior interpretation

Data Features
Feature	Description
sleep_hours	Hours of sleep last night
work_hours	Focused work/study duration
social_time	Minutes spent interacting socially
caffeine_intake	Cups of coffee/tea/energy drinks
steps	Walk/exercise activity
screen_time	Hours on computer/phone
water_intake	Hydration level (L/day)
mood	Morning self-rating (1–5)
noise_level	Environment sound level (1–10)
weather	Sunny / Rainy / Cloudy
stress_level	Target label (0–10)

Dataset file → stress_data.csv

Model Training

Two ML models were trained & compared:

Model	MAE ↓ (Error)	R² ↑ (Fit Score)
Linear Regression	0.1366	0.9773
Random Forest Regressor	0.3450	0.8613

Result:

Linear Regression performed best — the stress pattern in this dataset appears highly linear.

Key Insights from Feature Importance
Rank	            Feature	                           Influence
1	               social_time	                 Highest impact on stress
2	             steps (activity)	               Strong influence on stress response
3	                mood	                       Mental baseline matters a lot
4–6	  work_hours, sleep_hours, noise_level	   Direct contributors
7–9	     caffeine, screen_time, hydration          	Moderate effects
10–11   	weather factors	Minimal                    impact comparatively


Interpretation:

Stress appears more behavior-driven than environment-driven.
Social engagement, activity, mood & work balance are the strongest influencers.


Visual Outputs

 Correlation Heatmap
 Actual vs Predicted Plots
 Feature Importance Bar Graph
 Scatterplots of key variables vs stress


Future Enhancements

Upgrade	                             Value
Collect more days of data	     Increases model generalizability
Add XGBoost or LSTM	           Improved predictive performance
Convert to Classification	     Categorize stress → Low/Med/High
Build Streamlit App	           Real-time stress prediction UI
Deploy on web	                 Shareable live interactive model


Summary

This project models stress like a measurable, learnable system using data from daily habits — a real example of machine learning applied to personal wellness and behavior understanding. 

