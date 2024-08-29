# Money Matters: Exploring the State of Financial Well-Being in the U.S-Capstone Project
### Data Source: Data.gov (Consumer Financial Protection Bureau (CFPB), 2017).

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
  
#### Python Libraries used
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from scipy.stats import pearsonr

![image](https://github.com/user-attachments/assets/e9f37e89-7619-43b3-a4da-f1576b1163b5)

  
![Money-Matters-Exploring-the-State-of-Financial-Well-Being-in-the-U.S](images/Distribution%20of%20Financial%20Wellbeing.png)

Sample Size (N): 6,394
Mean: 56.08
Standard Deviation: 14.06
Minimum: 14
Median: 56
Maximum: 95
The histogram displays a normal distribution centered around the mean, with data points spread symmetrically around it
   
![Money-Matters-Exploring-the-State-of-Financial-Well-Being-in-the-U.S](images/Distribution%20of%20Financial%20Skills.png)

   Sample Size (N): 6,394
Mean: 50.78
Standard Deviation: 12.52
Minimum: 5
Median: 50
Maximum: 85
Shape: The histogram shows a distribution with a positive skew (skewness = 0.287), indicating a rightward tail.

![Money-Matters-Exploring-the-State-of-Financial-Well-Being-in-the-U.S](images/Financial%20well%20being%20by%20Financial%20Skills.png)
 Pearson r = 0.49 indicates a moderate correlation between financial skills and financial well-being.

 ![Money-Matters-Exploring-the-State-of-Financial-Well-Being-in-the-U.S](images/Average%20Financial%20Wellbeing%20by%20HousingMarketLosses.png)

On the belief that housing prices in the US can never go down, those who believe the statement is false have a higher average financial well-being score (56.7) 
compared to those who believe the statement is true. 

![Money-Matters-Exploring-the-State-of-Financial-Well-Being-in-the-U.S](images/Average%20Financial%20Wellbeing%20by%20LongTermReturns.png)
   
 In terms of long-term investment(10-20 years), stocks have the highest average financial well-being score (58.8), compared to bonds(52.7) and savings accounts (48.7).

![Money-Matters-Exploring-the-State-of-Financial-Well-Being-in-the-U.S](images/Financial%20Wellbeing%20by%20BondsInterestRates.png)

Regarding the impact of rising interest rates on bond prices: Those who expect bond prices to fall have the highest average financial well-being score (59.0), compared 
to those who expect bond prices to rise (55.9) and those who believe bond prices will stay the same (52.9).

![Money-Matters-Exploring-the-State-of-Financial-Well-Being-in-the-U.S](images/Average%20Financial%20Wellbeing%20by%20HousingMarketLosses.png)

 On the belief that housing prices in the US can never go down, those who believe the statement is false have a higher average financial well-being score (56.7) 
 compared to those who believe the statement is true. 

![Money-Matters-Exploring-the-State-of-Financial-Well-Being-in-the-U.S](images/Financial%20well%20being%20by%20Mortgage%20Interest%20Rate.png)
   
 Regarding the statement that a 15-year mortgage requires higher monthly payments but results in less total interest paid compared to a 30-year mortgage, those 
 who answered ‘True’ have a higher average financial well-being score (56.6) than those who answered ‘False’ (51.0).
  
![Money-Matters-Exploring-the-State-of-Financial-Well-Being-in-the-U.S](images/Average%20Financial%20Wellbeing%20by%20StocksVsBondsVolatility.png)
   
![Money-Matters-Exploring-the-State-of-Financial-Well-Being-in-the-U.S](images/Correlation%20Heatmap.png)
 Warm colors = High positive correlations
Cooler colors = High negative correlations
Neutral colors = No correlations


   
 ### Power BI Analysis
 
  - Demographic Analysis: Used Power BI to create visualizations and dashboards analyzing the impact of demographic factors such as household income, education level, age, generation, and ethnicity on financial well-being scores.
  - Dashboards: Created interactive dashboards to visualize and explore demographic data.

![Money-Matters-Exploring-the-State-of-Financial-Well-Being-in-the-U.S](images/Financial%20Wellbeing%20Power%20BI%20Dashboard.png)
   
   
 ### Machine Learning Model Development Process
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

![Money-Matters-Exploring-the-State-of-Financial-Well-Being-in-the-U.S](images/Perfoamnce%20Metric%20model.png)
 
Negative correlation between MSE(measures prediction error) and R^2(accuracy)
High MSE => Low R^2
Low MSE => High R^2


   

### Analysis and Insights
#### Demographic Factors (Power BI)
- Education Level: Higher education levels correlate with higher financial well-being scores.
- Race and Ethnicity: Variations suggest economic disparities, with Whites having higher scores and Black and Hispanic individuals having lower scores.
- Generational Differences: Older generations exhibit better financial well-being scores compared to younger generations.
- Age and Income: Higher household income and older age are associated with better financial well-being

#### Beliefs and Knowledge about Financial Concepts (Python)
- Financial Literacy: Greater understanding of financial concepts like investment risks and interest rates is linked to higher financial well-being scores.
- Knowledge of Financial Terms: Understanding mortgage terms and financial planning contributes to better financial outcomes.
#### Recommendations
1. **Enhance Financial Literacy Programs**:
  - Targeted Education: Focus on financial concepts for younger individuals and those with lower educational attainment.
  - Integrate into School Curriculums: Advocate for financial education from an early age.
2. **Promote Health and Well-Being Initiatives**:
  - Holistic Programs: Combine health and financial wellness programs.
  - Mental Health Support: Offer resources for stress management.
3. **Provide Tailored Financial Planning Resources**:
  - Customized Advice: Develop personalized financial advice based on demographic profiles.
  - Focus on Retirement Planning: Help younger generations learn from older generations’ practices.
4. **Encourage Long-Term Financial Stability**:
  - Promote Savings and Investment: Emphasize the importance of regular savings and investments.
  - Increase Awareness of Shared Financial Responsibilities: Provide resources on the benefits of shared responsibilities in marital relationships.









