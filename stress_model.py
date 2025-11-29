#!/usr/bin/env python
# coding: utf-8

# In[8]:


df = pd.read_csv("Downloads/stress_data.csv")


# In[11]:


df.info()
df.describe()


# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt

# Select only numeric columns
numeric_df = df.select_dtypes(include=['number'])

# Plot the correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# In[17]:


sns.scatterplot(data=df, x="sleep_hours", y="stress_level")
sns.scatterplot(data=df, x="work_hours", y="stress_level")
sns.scatterplot(data=df, x="screen_time", y="stress_level")
plt.show()


# In[23]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Drop only if columns exist
drop_cols = [col for col in ["stress_level", "date"] if col in df.columns]
X = df.drop(drop_cols, axis=1)

# Encode non-numeric columns
X = pd.get_dummies(X, drop_first=True)

# Target variable
y = df["stress_level"]  # Change this if the actual column name differs

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Preprocessing complete.")


# In[25]:


import pandas as pd

df = pd.read_csv("Downloads/stress_data.csv")
df.head()


# In[26]:


df.info()
df.describe().T


# In[27]:


print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
df.sample(3)


# In[29]:


# Keep only numeric columns
numeric_df = df.select_dtypes(include='number')

# Plot heatmap of correlations among numeric columns
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()


# In[30]:


features = ["sleep_hours", "work_hours", "screen_time", "steps", "caffeine_intake", "social_time", "water_intake"]

for col in features:
    plt.figure(figsize=(5,4))
    sns.scatterplot(data=df, x=col, y="stress_level")
    plt.title(f"{col} vs Stress Level")
    plt.show()


# In[31]:


plt.figure(figsize=(10,5))
sns.boxplot(data=df[["sleep_hours","work_hours","screen_time","stress_level"]])
plt.title("Value Distribution Overview")
plt.show()


# In[32]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("📌 Linear Regression Results:")
print("MAE:", mean_absolute_error(y_test, lr_pred))
print("R2 Score:", r2_score(y_test, lr_pred))


# In[33]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\n📌 Random Forest Results:")
print("MAE:", mean_absolute_error(y_test, rf_pred))
print("R2 Score:", r2_score(y_test, rf_pred))


# In[2]:


from xgboost import XGBRegressor
print("✅ XGBoost is ready!")


# In[8]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ✅ Load the data (this will now work!)
df = pd.read_csv("stress_data.csv")

# ✅ Drop columns if they exist
drop_cols = [col for col in ["stress_level", "date"] if col in df.columns]
X = df.drop(drop_cols, axis=1)
y = df["stress_level"]

# ✅ Encode categorical features (if any)
X = pd.get_dummies(X, drop_first=True)

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:





# In[11]:


from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Create and train the model
xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5)
xgb.fit(X_train, y_train)

# Make predictions
xgb_pred = xgb.predict(X_test)

# Evaluate the model
print("\n📌 XGBoost Results:")
print("MAE:", mean_absolute_error(y_test, xgb_pred))
print("R2 Score:", r2_score(y_test, xgb_pred))


# In[14]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)


# In[15]:


import matplotlib.pyplot as plt

plt.figure(figsize=(6,5))
plt.scatter(y_test, rf_pred, alpha=0.7)
plt.xlabel("Actual Stress Level")
plt.ylabel("Predicted Stress Level")
plt.title("Random Forest Prediction Performance")
plt.show()


# In[16]:


import numpy as np

plt.figure(figsize=(6,5))
plt.scatter(y_test, rf_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal line
plt.xlabel("Actual Stress Level")
plt.ylabel("Predicted Stress Level")
plt.title("Random Forest Prediction Performance")
plt.show()


# In[17]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)


# In[18]:


lr_pred = lr.predict(X_test)


# In[19]:


from sklearn.metrics import mean_absolute_error, r2_score

lr_mae = mean_absolute_error(y_test, lr_pred)
lr_r2  = r2_score(y_test, lr_pred)

print("📌 Linear Regression Performance")
print("MAE:", lr_mae)
print("R² Score:", lr_r2)


# In[20]:


import matplotlib.pyplot as plt

plt.figure(figsize=(6,5))
plt.scatter(y_test, lr_pred, alpha=0.7, color="blue")
plt.plot([0,10],[0,10], color="red") # perfect prediction line
plt.xlabel("Actual Stress Level")
plt.ylabel("Predicted Stress Level")
plt.title("Linear Regression: Actual vs Predicted Stress")
plt.show()


# In[21]:


feature_names = X.columns


# In[22]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print("📌 Random Forest Performance")
print("MAE:", rf_mae)
print("R² Score:", rf_r2)


# In[23]:


import matplotlib.pyplot as plt

plt.figure(figsize=(6,5))
plt.scatter(y_test, rf_pred, alpha=0.7)
plt.plot([0,10], [0,10], color="red")  # perfect line
plt.xlabel("Actual Stress Level")
plt.ylabel("Predicted Stress Level")
plt.title("Random Forest: Actual vs Predicted Stress")
plt.show()


# In[24]:


import numpy as np
import pandas as pd

importances = rf.feature_importances_
importance_series = pd.Series(importances, index=feature_names).sort_values(ascending=True)

plt.figure(figsize=(8,6))
importance_series.plot(kind="barh")
plt.title("Feature Importances - Random Forest")
plt.xlabel("Importance Score")
plt.show()

importance_series


# In[25]:


# ============================
# 1. Train Random Forest
# ============================

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)


# ============================
# 2. Evaluation
# ============================

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2  = r2_score(y_test, rf_pred)

print("RANDOM FOREST RESULTS")
print("MAE:", rf_mae)
print("R² Score:", rf_r2)

print("\n LINEAR REGRESSION (for comparison)")
print("MAE:", lr_mae)
print("R² Score:", lr_r2)


# ============================
# 3. Actual vs Predicted Plot
# ============================

import matplotlib.pyplot as plt

plt.figure(figsize=(6,5))
plt.scatter(y_test, rf_pred, alpha=0.7, color="teal")
plt.plot([0,10], [0,10], color="red")   # perfect prediction line
plt.xlabel("Actual Stress Level")
plt.ylabel("Predicted Stress Level")
plt.title("Random Forest: Actual vs Predicted Stress Level")
plt.show()


# ============================
# 4. Feature Importance Plot
# ============================

import pandas as pd
import numpy as np

feature_importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=True)

plt.figure(figsize=(8,6))
feature_importances.plot(kind="barh", color="purple")
plt.title("Feature Importances (Random Forest)")
plt.xlabel("Importance Score")
plt.show()

feature_importances


# In[ ]:


### 🔍 Key Findings from Feature Importance
- Social interaction has the highest impact on stress levels.
- Mood and physical activity (steps) are strong predictors.
- Work hours and sleep hours show nearly equal influence.
- Caffeine, screen time, and noise moderately raise stress.
- Weather conditions have minimal effect compared to habits.


# In[ ]:




