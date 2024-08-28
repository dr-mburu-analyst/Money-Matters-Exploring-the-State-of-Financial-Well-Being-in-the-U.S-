# Money Matters: Exploring the State of Financial Well-Being in the U.S-Capstone Project
### Overview
- This capstone project explores the factors affecting financial well-being and develops a machine learning model to predict financial well-being scores.
- The analysis involves using Python for data processing, exploratory data analysis, and visualization, and Power BI for analyzing demographic factors.
### Guiding Questions
1. How do demographic factors such as household income, education level, age, generation, and ethnicity affect an individual’s financial well-being score?
2. How do individuals’ beliefs and knowledge about financial concepts, such as investment risks, interest rates, and mortgage terms, impact their financial well-being scores?
3. How can we develop a machine learning model that integrates both demographic factors and individuals’ beliefs and knowledge about financial concepts to accurately predict financial well-being scores?
4. ### Approach
5. #### Data Processing and Analysis with Python
   - Data Processing: Used Python for data cleaning and preparation.
   - Exploratory Data Analysis (EDA): Conducted exploratory analysis to identify patterns and relationships.
   - Visualization: Created visualizations to understand how beliefs and knowledge about financial concepts impact financial well-being scores.
 #### Power BI Analysis
  - Demographic Analysis: Used Power BI to create visualizations and dashboards analyzing the impact of demographic factors such as household income, education level, age, generation, and ethnicity on financial well-being scores.
  - Dashboards: Created interactive dashboards to visualize and explore demographic data.

 #### Machine Learning Model Development Process
 ##### Data Splitting:
   - from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
##### Feature Scaling:
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
##### Model Selection and Training:
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor()
model.fit(X_train, y_train)
##### Model Tuning:
###### Hyperparameter Tuning:
from sklearn.model_selection import GridSearchCV

param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'n_estimators': [50, 100, 150]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_

Best Settings:
Learning Rate: 0.1
Max Depth: 3
Number of Estimators: 100
###### Cross-Validation: 
Ensured the model's generalizability
##### Model Evaluation:
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)










