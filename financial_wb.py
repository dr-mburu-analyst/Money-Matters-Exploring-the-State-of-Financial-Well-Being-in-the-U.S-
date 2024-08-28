# Start by importing the necessary libraries for data analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from scipy.stats import pearsonr

# Loading the Loan Data and converting into dataframe

financial_wb_df = pd.read_csv('NFWBS_PUF_2016_data.csv')

# Initial exploration of the dataset to understand the structure, types of data and quality

print("financial_wb_df.head")
print(financial_wb_df.head())
print("")

# print the last 5 rows
print("financial_wb_df.tail")
print(financial_wb_df.tail())
print("")

# print the info
print("financial_wb_df.info")
print(financial_wb_df.info())
print("")

# Print column names
print("Columns in the DataFrame:")
print(financial_wb_df.columns)
print("")

# Data Cleaning: Handling Duplicates
print("financial_wb_df.duplicated().any()")
print(financial_wb_df.duplicated().any())
print("")

# check for data shape
print("financial_wb_df.shape")
print(financial_wb_df.shape)
print("")

# Determining Relevent columns to use based on the projects goal which is to identify how various demographic factors(age, gender, education level, occupation, income, etc.), 
# financial well being indicators(income levels, savings, debt, expenditure, investments, etc.) and other knowledge contribute to an individual's financial well-being
financial_wb_df =financial_wb_df [['FWBscore', 'FSscore', 'PPINCIMP', 'PPMARIT','KHKNOWL1','KHKNOWL2','KHKNOWL3','KHKNOWL4','KHKNOWL5','KHKNOWL6','KHKNOWL7','KHKNOWL8','KHKNOWL9',
'HEALTH','RETIRE','agecat','generation','PPEDUC','PPETHM','PPGENDER','PPREG9','SWB_1']]
print(financial_wb_df.head())

# Rename columns to improve readability and to facilitate communication
financial_wb_df.rename(columns={'PPINCIMP':'Income',
                                'FWBscore':'Financial_wellbeing',
                                'FSscore':'Financial_skill',
                        'KHKNOWL1':'LongTermReturns',
                        'KHKNOWL2':'StocksVsBondsVolatility',
                        'KHKNOWL3':'DiversificationBenefits',
                        'KHKNOWL4':'StockMarketLosses',
                        'KHKNOWL5':'LifeInsuranceKnowledge',
                        'KHKNOWL6':'HousingMarketLosses',
                        'KHKNOWL7':'CreditCardPayments',
                        'KHKNOWL8':'BondsInterestRates',
                        'KHKNOWL9':'MortgageTermInterest',
                        'agecat':'Age Category',
                        'PPEDUC':'Education',
                        'PPETHM':'Ethnicity',
                        'PPGENDER':'Gender',
                        'PPREG9':'Region',
                        'SWB_1':'LifeSatisfaction'},
                        inplace=True)
print(financial_wb_df.columns)

# Further Cleaning and Preparation
# After narrowing down the relevant columns, I performed more targeted cleaning on these columns. 
# For example, I identified special values in my data that were placeholders for missing or non-standard responses ('-1 for refused','-4 for no response written', and' 99 for Prefer not to say') 
# I converted the special values to Nan (Not a Number) to standardize them as missing values. This was important because most imputation methods and analysis techniques recognize NaN
# as an indicator of missing data

financial_wb_df.replace([-1, -4, 99], np.nan, inplace=True)
# Check for and remove missing values from columns of interest
# Verify replacement
print((financial_wb_df == -1).sum())
print((financial_wb_df == -4).sum())
print((financial_wb_df == 99).sum())

print(len(financial_wb_df['RETIRE']))
print(len(financial_wb_df['Financial_wellbeing']))


# #Once the special values were replaces with NaN, I used an imputation method to fill these missing values.
#  We can replace the missing values of the column 'Financial_wellbeing' with the mean of the column 'Financial_wellbeing'  
# # using the method replace(). Don't forget to set the inplace parameter to True

mean=financial_wb_df['Financial_wellbeing'].mean()
financial_wb_df['Financial_wellbeing'] = financial_wb_df['Financial_wellbeing'].fillna(mean)

mean=financial_wb_df['Financial_skill'].mean()
financial_wb_df['Financial_skill'] = financial_wb_df['Financial_skill'].fillna(mean)

mean=financial_wb_df['LongTermReturns'].mean()
financial_wb_df['LongTermReturns'] = financial_wb_df['LongTermReturns'].fillna(mean)

mean=financial_wb_df['StocksVsBondsVolatility'].mean()
financial_wb_df['StocksVsBondsVolatility'] = financial_wb_df['StocksVsBondsVolatility'].fillna(mean)

mean=financial_wb_df['DiversificationBenefits'].mean()
financial_wb_df['DiversificationBenefits'] = financial_wb_df['DiversificationBenefits'].fillna(mean)

mean=financial_wb_df['StockMarketLosses'].mean()
financial_wb_df['StockMarketLosses'] = financial_wb_df['StockMarketLosses'].fillna(mean)

mean=financial_wb_df['LifeInsuranceKnowledge'].mean()
financial_wb_df['LifeInsuranceKnowledge'] = financial_wb_df['LifeInsuranceKnowledge'].fillna(mean)

mean=financial_wb_df['HousingMarketLosses'].mean()
financial_wb_df['HousingMarketLosses'] = financial_wb_df['HousingMarketLosses'].fillna(mean)

mean=financial_wb_df['CreditCardPayments'].mean()
financial_wb_df['CreditCardPayments'] = financial_wb_df['CreditCardPayments'].fillna(mean)

mean=financial_wb_df['BondsInterestRates'].mean()
financial_wb_df['BondsInterestRates'] = financial_wb_df['BondsInterestRates'].fillna(mean)

mean=financial_wb_df['MortgageTermInterest'].mean()
financial_wb_df['MortgageTermInterest'] = financial_wb_df['MortgageTermInterest'].fillna(mean)

mean=financial_wb_df['HEALTH'].mean()
financial_wb_df['HEALTH'] = financial_wb_df['HEALTH'].fillna(mean)

mean=financial_wb_df['RETIRE'].mean()
financial_wb_df['RETIRE'] = financial_wb_df['RETIRE'].fillna(mean)

mean=financial_wb_df['LifeSatisfaction'].mean()
financial_wb_df['LifeSatisfaction'] = financial_wb_df['LifeSatisfaction'].fillna(mean)

# Verify and Validate
# After the imputation method, I verified that the missing values had been filled appropriately and that the data was now consistent
print(financial_wb_df.isnull().sum())

# Exploratory Data Analysis (EDA):
# Visualize Relationships: I Used different visualization techniques to explore relationships between demographic factors and financial well-being. 
# I used Scatter plots, bar charts, boxplots and heatmaps. I also Looked for patterns or trends that might indicate how different factors contribute to financial well-being.


# Compute Pearson correlation coefficient
pearson_corr, _ = pearsonr(financial_wb_df['Financial_skill'], financial_wb_df['Financial_wellbeing'])

# Create the scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=financial_wb_df, x='Financial_skill', y='Financial_wellbeing', color='blue')

# Add a regression line (optional)
sns.regplot(data=financial_wb_df, x='Financial_skill', y='Financial_wellbeing', scatter=False, color='red')

# Add the Pearson r value to the plot
plt.text(x=0.05, y=0.95, s=f'Pearson r = {pearson_corr:.2f}', 
        transform=plt.gca().transAxes, 
        fontsize=12, verticalalignment='top', color='black')

# Set titles and labels
plt.title('Scatterplot of Financial Wellbeing vs. Financial Skills')
plt.xlabel('Financial Skills')
plt.ylabel('Financial Wellbeing')

# Show the plot
plt.show()

# Plotting the boxplot of Income vs Financial_wellbeing

# Create the mapping dictionary
income_mapping = {
1: 'Less than 20k',
2: '20k-29k',
3: '30k-39k',
4: '40k-49k',
5: '50k-59k',
6: '60k-69k',
7: '75k-99k',
8: '100k-149k',
9: '150k & above'
}
# Define the order of categories
income_order = [
'Less than 20k',
'20k-29k',
'30k-39k',
'40k-49k',
'50k-59k',
'60k-69k',
'75k-99k',
'100k-149k',
'150k & above'
]
# Apply the mapping to the 'Income' column
financial_wb_df['Income'] = financial_wb_df['Income'].map(income_mapping)

# Convert the 'Income' column to categorical with the specified order
financial_wb_df['Income'] = pd.Categorical(financial_wb_df['Income'], categories=income_order, ordered=True)


# Verify the mapping
print(financial_wb_df['Income'].unique())

plt.figure(figsize=(10, 6))
sns.boxplot(data = financial_wb_df, x='Income', y='Financial_wellbeing', palette='viridis')
plt.title('Box Plot of Financial_wellbeing by Income')
plt.xlabel('Income')
plt.ylabel('Financial_wellbeing')
plt.xticks(rotation=45)  # Rotate x-axis labels if needed for readability
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# plotting a boxplot of LifeSatisfaction vs Financial Wellbeing

LifeSatisfaction_mapping = {
1: 'Strongly Disagree',
2: 'Disagree',
3: 'Disagree Slightly',
4: 'Neutral',
5: 'Agree Slightly',
6: 'Agree',
7: 'Strongly Agree' 
}

# # Define the order of categories
LifeSatisfaction_order = [
'Strongly Disagree',
'Disagree',
'Disagree Slightly',
'Neutral',
'Agree Slightly',
'Agree',
'Strongly Agree'  
]

# Apply the mapping to the 'LifeSatisfaction' column
financial_wb_df['LifeSatisfaction'] = financial_wb_df['LifeSatisfaction'].map(LifeSatisfaction_mapping)

# Convert the 'LifeSatisfaction' column to categorical with the specified order
financial_wb_df['LifeSatisfaction'] = pd.Categorical(financial_wb_df['LifeSatisfaction'], categories=LifeSatisfaction_order, ordered=True)

sns.boxplot(data=financial_wb_df, x='LifeSatisfaction', y='Financial_wellbeing', palette='viridis')
plt.title('Boxplot of LifeSatisfaction vs Financial Wellbeing')
plt.xlabel('LifeSatisfaction')
plt.ylabel('Financial Wellbeing')
plt.xticks(rotation=45)  # Rotate x-axis labels if needed for readability
# plt.xticks(rotation=45)
plt.show()


# plotting a boxplot of age category and financial well being
Age_Category_mapping = {
1: '18-24',
2: '25-34',
3: '35-44',
4: '45-54',
5: '55-61',
6: '62-69',
7: '70-74',
8: '75+' 
}

# # Define the order of categories
Age_Category_order = [
'18-24',
'25-34',
'35-44',
'45-54',
'55-61',
'62-69',
'70-74',
'75+'   
]

# Apply the mapping to the 'LifeSatisfaction' column
financial_wb_df['Age Category'] = financial_wb_df['Age Category'].map(Age_Category_mapping)

# # Convert the 'LifeSatisfaction' column to categorical with the specified order
financial_wb_df['Age Category'] = pd.Categorical(financial_wb_df['Age Category'], categories=Age_Category_order, ordered=True)

sns.boxplot(data=financial_wb_df, x='Age Category', y='Financial_wellbeing', palette='viridis')
plt.title('Boxplot of Age Category vs Financial Wellbeing')
plt.xlabel('Age Category')
plt.ylabel('Financial Wellbeing')
plt.xticks(rotation=45)  # Rotate x-axis labels if needed for readability
# plt.xticks(rotation=45)
plt.show()



# # plotting a barplot of Financial_wellbeing by MortgageTermInterest

MortgageTermInterest_mapping = {
1: 'True',
2: 'False'
}

MortgageTermInterest_order = [
'True',
'False'
]

# # Apply the mapping to the 'LifeSatisfaction' column
financial_wb_df['MortgageTermInterest'] = financial_wb_df['MortgageTermInterest'].map(MortgageTermInterest_mapping)

# # # Convert the 'LifeSatisfaction' column to categorical with the specified order
financial_wb_df['MortgageTermInterest'] = pd.Categorical(financial_wb_df['MortgageTermInterest'], categories=MortgageTermInterest_order, ordered=True)
# Aggregating data to calculate mean Financial Wellbeing for each LongTermReturns category
grouped_df = financial_wb_df.groupby('MortgageTermInterest')['Financial_wellbeing'].mean().reset_index()

ax = sns.barplot(data=grouped_df, x='MortgageTermInterest', y='Financial_wellbeing', palette='plasma')
# Add data labels to each bar
for p in ax.patches:
height = p.get_height()
ax.annotate(f'{height:.2f}',  # Format the label
                (p.get_x() + p.get_width() / 2., height),  # Position the label
                ha='center', va='center',  # Horizontal and vertical alignment
                xytext=(0, 5),  # Offset for the label
                textcoords='offset points')
plt.title('Average of Financial Wellbeing by MortgageTermInterest')
plt.xlabel('MortgageTermInterest')
plt.ylabel('Financial_wellbeing')
# plt.xticks(rotation=45)  # Rotate x-axis labels if needed for readability
# plt.xticks(rotation=45)
plt.show()


# # Plotting a pie chart of LifeInsuranceKnowledge and Financial wellbeing
plt.figure(figsize=(8, 8))
plt.pie(grouped_df['Financial_wellbeing'], labels=grouped_df['LifeInsuranceKnowledge'], autopct='%1.1f%%', colors=sns.color_palette('viridis', len(grouped_df)))
plt.title('Average Financial Wellbeing by LifeInsuranceKnowledge')
plt.show()



# Create the bar plot of Financial well being by LifeInsuranceKnowledge
LifeInsuranceKnowledge_mapping = {
1: 'True',
2: 'False'
}

LifeInsuranceKnowledge_order = [
'True',
'False'
]

# Apply the mapping to the 'LifeInsuranceKnowledge' column
financial_wb_df['LifeInsuranceKnowledge'] = financial_wb_df['LifeInsuranceKnowledge'].map(LifeInsuranceKnowledge_mapping)

# # # # Convert the 'LifeInsuranceKnowledge' column to categorical with the specified order
financial_wb_df['LifeInsuranceKnowledge'] = pd.Categorical(financial_wb_df['LifeInsuranceKnowledge'], categories=LifeInsuranceKnowledge_order, ordered=True)

# Apply the mapping to the 'LifeInsuranceKnowledge' column
# financial_wb_df['LifeInsuranceKnowledge'] = financial_wb_df['LifeInsuranceKnowledge'].map(LifeInsuranceKnowledge_mapping)


# Aggregating data to calculate mean Financial Wellbeing for each  MortgageTermInterest category
grouped_df = financial_wb_df.groupby('LifeInsuranceKnowledge')['Financial_wellbeing'].mean().reset_index()

plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
ax = sns.barplot(data=grouped_df, x='LifeInsuranceKnowledge', y='Financial_wellbeing', palette='plasma')

# Add data labels to each bar
for p in ax.patches:
height = p.get_height()
ax.annotate(f'{height:.1f}',  # Format the label to one decimal place
                (p.get_x() + p.get_width() / 2., height),  # Position the label
                ha='center', va='center',  # Horizontal and vertical alignment
                xytext=(0, 5),  # Offset for the label
                textcoords='offset points')

# Set titles and labels
plt.title('Average Financial Wellbeing by Life Insurance Knowledge')
plt.xlabel('Life Insurance Knowledge')
plt.ylabel('Financial Wellbeing')

# Show the plot
plt.show()



# # Create the bar plot of Financial well being by LongTermReturns
LongTermReturns_mapping = {
1: 'Savings Accounts',
2: 'Bonds',
3: 'Stocks'
}

LongTermReturns_order = ['Savings Accounts','Bonds','Stocks']

# Apply the mapping to the 'LifeInsuranceKnowledge' column
financial_wb_df['LongTermReturns'] = financial_wb_df['LongTermReturns'].map(LongTermReturns_mapping)

# # # # Convert the 'LongTermReturns' column to categorical with the specified order
financial_wb_df['LongTermReturns'] = pd.Categorical(financial_wb_df['LongTermReturns'], categories=LongTermReturns_order, ordered=True)

# Apply the mapping to the 'LongTermReturns' column
# financial_wb_df['LongTermReturns'] = financial_wb_df['LongTermReturns'].map(LongTermReturns_mapping)


# Aggregating data to calculate mean Financial Wellbeing for each LongTermReturns category
grouped_df = financial_wb_df.groupby('LongTermReturns')['Financial_wellbeing'].mean().reset_index()

plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
ax = sns.barplot(data=grouped_df, x='LongTermReturns', y='Financial_wellbeing', palette='plasma')

# Add data labels to each bar
for p in ax.patches:
height = p.get_height()
ax.annotate(f'{height:.1f}',  # Format the label to one decimal place
                (p.get_x() + p.get_width() / 2., height),  # Position the label
                ha='center', va='center',  # Horizontal and vertical alignment
                xytext=(0, 5),  # Offset for the label
                textcoords='offset points')

# Set titles and labels
plt.title('Average Financial Wellbeing by LongTermReturns')
plt.xlabel('Long Term Returns')
plt.ylabel('Financial Wellbeing')

# Show the plot
plt.show()



# Create the bar plot of Financial well being by BondsInterestRates
BondsInterestRates_mapping = {
1: 'Rise',
2: 'Fall',
3: 'Stay Same'
}

BondsInterestRates_order = ['Rise','Fall','Stay Same']

# Apply the mapping to the 'BondsInterestRates' column
financial_wb_df['BondsInterestRates'] = financial_wb_df['BondsInterestRates'].map(BondsInterestRates_mapping)

# # # # Convert the 'BondsInterestRates' column to categorical with the specified order
financial_wb_df['BondsInterestRates'] = pd.Categorical(financial_wb_df['BondsInterestRates'], categories=BondsInterestRates_order, ordered=True)

# Apply the mapping to the 'BondsInterestRates' column
# financial_wb_df['BondsInterestRates'] = financial_wb_df['BondsInterestRates'].map(BondsInterestRates_mapping)


# Aggregating data to calculate mean Financial Wellbeing for each BondsInterestRates category
grouped_df = financial_wb_df.groupby('BondsInterestRates')['Financial_wellbeing'].mean().reset_index()

plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
ax = sns.barplot(data=grouped_df, x='BondsInterestRates', y='Financial_wellbeing', palette='plasma')

# Add data labels to each bar
for p in ax.patches:
height = p.get_height()
ax.annotate(f'{height:.1f}',  # Format the label to one decimal place
                (p.get_x() + p.get_width() / 2., height),  # Position the label
                ha='center', va='center',  # Horizontal and vertical alignment
                xytext=(0, 5),  # Offset for the label
                textcoords='offset points')

# Set titles and labels
plt.title('Average Financial Wellbeing by BondsInterestRates')
plt.xlabel('Bonds Interest Rates')
plt.ylabel('Financial Wellbeing')

# Show the plot
plt.show()



# Create the bar plot of Financial well being by DiversificationBenefits
DiversificationBenefits_mapping = {
1: 'Increase',
2: 'Decrease',
3: 'Stay Same'
}

DiversificationBenefits_order = ['Increase','Decrease','Stay Same']

# Apply the mapping to the 'DiversificationBenefits' column
financial_wb_df['DiversificationBenefits'] = financial_wb_df['DiversificationBenefits'].map(DiversificationBenefits_mapping)

# # # # Convert the 'DiversificationBenefits' column to categorical with the specified order
financial_wb_df['DiversificationBenefits'] = pd.Categorical(financial_wb_df['DiversificationBenefits'], categories=DiversificationBenefits_order, ordered=True)

# Apply the mapping to the 'BondsInterestRates' column
# financial_wb_df['BondsInterestRates'] = financial_wb_df['BondsInterestRates'].map(BondsInterestRates_mapping)


# Aggregating data to calculate mean Financial Wellbeing for each BondsInterestRates category
grouped_df = financial_wb_df.groupby('DiversificationBenefits')['Financial_wellbeing'].mean().reset_index()

plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
ax = sns.barplot(data=grouped_df, x='DiversificationBenefits', y='Financial_wellbeing', palette='plasma')

# Add data labels to each bar
for p in ax.patches:
height = p.get_height()
ax.annotate(f'{height:.1f}',  # Format the label to one decimal place
                (p.get_x() + p.get_width() / 2., height),  # Position the label
                ha='center', va='center',  # Horizontal and vertical alignment
                xytext=(0, 5),  # Offset for the label
                textcoords='offset points')

# Set titles and labels
plt.title('Average Financial Wellbeing by DiversificationBenefits')
# plt.xlabel('Diversification Benefits')
plt.ylabel('Financial Wellbeing')

# Show the plot
plt.show()


# Create the bar plot of Financial well being byHousingMarketLosses
HousingMarketLosses_mapping = {
1: 'True',
2: 'False'
}

HousingMarketLosses_order = ['True','False']

# Apply the mapping to the 'DiversificationBenefits' column
financial_wb_df['HousingMarketLosses'] = financial_wb_df['HousingMarketLosses'].map(HousingMarketLosses_mapping)

# # # # Convert the 'HousingMarketLosses' column to categorical with the specified order
financial_wb_df['HousingMarketLosses'] = pd.Categorical(financial_wb_df['HousingMarketLosses'], categories=HousingMarketLosses_order, ordered=True)

# Apply the mapping to the 'BondsInterestRates' column
# financial_wb_df['BondsInterestRates'] = financial_wb_df['BondsInterestRates'].map(BondsInterestRates_mapping)


# Aggregating data to calculate mean Financial Wellbeing for each BondsInterestRates category
grouped_df = financial_wb_df.groupby('HousingMarketLosses')['Financial_wellbeing'].mean().reset_index()

plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
ax = sns.barplot(data=grouped_df, x='HousingMarketLosses', y='Financial_wellbeing', palette='plasma')

# Add data labels to each bar
for p in ax.patches:
height = p.get_height()
ax.annotate(f'{height:.1f}',  # Format the label to one decimal place
                (p.get_x() + p.get_width() / 2., height),  # Position the label
                ha='center', va='center',  # Horizontal and vertical alignment
                xytext=(0, 5),  # Offset for the label
                textcoords='offset points')

# Set titles and labels
plt.title('Average Financial Wellbeing by HousingMarketLosses')
# plt.xlabel('Diversification Benefits')
plt.ylabel('Financial Wellbeing')

# Show the plot
plt.show()



# Create the bar plot of Financial well being by StockMarketLosses
StockMarketLosses_mapping = {
1: 'True',
2: 'False'
}

StockMarketLosses_order = ['True','False']

# Apply the mapping to the 'StockMarketLosses' column
financial_wb_df['StockMarketLosses'] = financial_wb_df['StockMarketLosses'].map(StockMarketLosses_mapping)

# # # # Convert the 'StockMarketLosses' column to categorical with the specified order
financial_wb_df['StockMarketLosses'] = pd.Categorical(financial_wb_df['StockMarketLosses'], categories=StockMarketLosses_order, ordered=True)

# Apply the mapping to the 'StockMarketLosses' column
# financial_wb_df['BondsInterestRates'] = financial_wb_df['BondsInterestRates'].map(BondsInterestRates_mapping)


# Aggregating data to calculate mean Financial Wellbeing for each StockMarketLosses category
grouped_df = financial_wb_df.groupby('StockMarketLosses')['Financial_wellbeing'].mean().reset_index()

plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
ax = sns.barplot(data=grouped_df, x='StockMarketLosses', y='Financial_wellbeing', palette='plasma')

# Add data labels to each bar
for p in ax.patches:
height = p.get_height()
ax.annotate(f'{height:.1f}',  # Format the label to one decimal place
                (p.get_x() + p.get_width() / 2., height),  # Position the label
                ha='center', va='center',  # Horizontal and vertical alignment
                xytext=(0, 5),  # Offset for the label
                textcoords='offset points')

# Set titles and labels
plt.title('Average Financial Wellbeing by StockMarketLosses')
# plt.xlabel('StockMarketLosses')
plt.ylabel('Financial Wellbeing')

# Show the plot
plt.show()


# Create the bar plot of Financial well being by StocksVsBondsVolatility
StocksVsBondsVolatility_mapping = {
1: 'True',
2: 'False'
}

StocksVsBondsVolatility_order = ['True','False']

# Apply the mapping to the 'StocksVsBondsVolatility' column
financial_wb_df['StocksVsBondsVolatility'] = financial_wb_df['StocksVsBondsVolatility'].map(StocksVsBondsVolatility_mapping)

# # # # Convert the 'StocksVsBondsVolatility' column to categorical with the specified order
financial_wb_df['StocksVsBondsVolatility'] = pd.Categorical(financial_wb_df['StocksVsBondsVolatility'], categories=StocksVsBondsVolatility_order, ordered=True)

# Apply the mapping to the 'StocksVsBondsVolatility' column
# financial_wb_df['BondsInterestRates'] = financial_wb_df['BondsInterestRates'].map(BondsInterestRates_mapping)


# Aggregating data to calculate mean Financial Wellbeing for each StocksVsBondsVolatility category
grouped_df = financial_wb_df.groupby('StocksVsBondsVolatility')['Financial_wellbeing'].mean().reset_index()

plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
ax = sns.barplot(data=grouped_df, x='StocksVsBondsVolatility', y='Financial_wellbeing', palette='plasma')

# Add data labels to each bar
for p in ax.patches:
height = p.get_height()
ax.annotate(f'{height:.1f}',  # Format the label to one decimal place
                (p.get_x() + p.get_width() / 2., height),  # Position the label
                ha='center', va='center',  # Horizontal and vertical alignment
                xytext=(0, 5),  # Offset for the label
                textcoords='offset points')

# Set titles and labels
plt.title('Average Financial Wellbeing by StocksVsBondsVolatility')
# plt.xlabel('StockMarketLosses')
plt.ylabel('Financial Wellbeing')

# Show the plot
plt.show()




# # Plotting the scatter plot of financialwellbeing and LongTermReturns
plt.figure(figsize=(10, 6))
plt.scatter(financial_wb_df['LongTermReturns'], financial_wb_df['Financial_wellbeing'], color='blue', marker='o')
plt.title('Scatter Plot of LongTermReturns vs Financial_wellbeing')
plt.xlabel('LongTermReturns')
plt.ylabel('Financial_wellbeing')
plt.grid(True)
plt.show()


# plot a Correlation heatmap
# Create a heatmap to visually represent the correlation between variables.

# Convert boolean columns to numeric
financial_wb_df = financial_wb_df.applymap(lambda x: 1 if x == 'True' else (0 if x == 'False' else x))

# Select only numeric columns
numeric_df = financial_wb_df.select_dtypes(include=[float, int])

# Compute the correlation matrix
correlation_matrix = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(financial_wb_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# Plot a histogram of Financial_wellbeing
plt.figure(figsize=(10, 6))
plt.hist(financial_wb_df['Financial_skill'], bins=30, color='green', edgecolor='black')
plt.title('Distribution of Financial Skills')
# plt.xlabel('Financial Wellbeing')
plt.ylabel('Frequency')
plt.show()

print(financial_wb_df.describe())


# # Based on the correlation heatmap
# # High positive correlations are typically represented by warmer colors (like red or orange).
# # High negative correlations are represented by cooler colors (like blue).
# # Little to no correlation is often shown in neutral colors (like white or light gray).


# Now we have obtained some preliminary insights about how each important variable would affect the success rate, we can now select
# the features that will be used in success prediction in the future  model. 

# Train and evaluate a Random Forest model for predicting a persons financial_wellbeing based on these features.

Feature selection: selected columns (features) from the dataset to be used for training the model.
features = financial_wb_df[['Financial_skill','Income','PPMARIT','LongTermReturns','HEALTH','RETIRE','Age Category','Education','LifeSatisfaction','generation',
                        'StocksVsBondsVolatility','DiversificationBenefits','StockMarketLosses','LifeInsuranceKnowledge','HousingMarketLosses','CreditCardPayments',
                        'BondsInterestRates','MortgageTermInterest','Ethnicity','Gender','Region']]
target ='Financial_Wellbeing'

X = financial_wb_df[['Financial_skill','Income','PPMARIT','LongTermReturns','HEALTH','RETIRE','Age Category','Education','LifeSatisfaction','generation',
                'StocksVsBondsVolatility','DiversificationBenefits','StockMarketLosses','LifeInsuranceKnowledge','HousingMarketLosses','CreditCardPayments',
                'BondsInterestRates','MortgageTermInterest','Ethnicity','Gender','Region']]
y = financial_wb_df['Financial_wellbeing']


# Model Creation and Evaluation
# Create and evaluate the model

# set up the training model
# Data Splitting: Split the dataset into training (X_train, y_train) and testing (X_test, y_test) sets using train_test_split.
# The 80% of the data is used for training, and 20% is used for testing
# random state 0 ensures the data split is the same every time you run the code

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Feature scaling: Normalize or standardize the data to ensure all features are on a similar scale.
# initialize and fit the scaler
# Normalize the features of the dataset using 'StandardScaler' from 'sklearn.preprocessing' module
#  by removing the mean and scaling to unit variance
scaler= StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data
X_test_scaled = scaler.transform(X_test)

# Build and  and train the model: Choose a machine learning model. train(fit the model to the training data to learn patterns and relationships)
# model = RandomForestRegressor(n_estimators=100, random_state=0)
# model.fit(X_train_scaled, y_train)

Tune the model: Hyperparameter tuning (adjusting the model's settings to improve its performance using techniques such as GridSearchCV; 
Cross-validation(to assess the models performance across different subsets of the training data to ensure it generalizes well))

# Define parameter grid
param_grid = {
'n_estimators': [50, 100, 150, 200],
'learning_rate': [0.01, 0.1, 0.2],
'max_depth': [3, 4, 5]
}

model = GradientBoostingRegressor(random_state=0)

# Initialize GridSearchCV
grid_search =GridSearchCV(model, param_grid, cv=5, scoring='r2')

# Fit GridSearchCV
grid_search.fit(X_train_scaled, y_train)

# Best parameters and best score


# Get the best model from GridSearchCV
best_model = grid_search.best_estimator_

# # Perform cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
# print(f'Cross-validated R-squared scores: {cv_scores}')
# print(f'Mean cross-validated R-squared: {cv_scores.mean()}')
# Make predictions

# Evaluate the model(use the test set to evaluate how well the model performs to unseen data). Calculate metrics such as Mean Squared Error (MSE) and R-squared Error (R^2)
#tO UNDERSTAND  how accurately the model predicts financial wellbeing
y_pred = best_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 =r2_score(y_test,y_pred)

print(f'Best parameters: {grid_search.best_params_}')
print(f'Best R-squared score: {grid_search.best_score_}')
print(f'Test Mean squared error:{mse:.2f}')
print(f'Test R-squared:{r2:.2f}')



# print(financial_wb_df.describe())

# Data to plot
metrics = {
'Mean Squared Error': 100.67,
'R-squared': 0.48
}

# Create a line plot
plt.figure(figsize=(10, 6))

# Plot each metric
plt.plot(metrics.keys(), metrics.values(), marker='o', linestyle='-', color='b')

# Add data labels to each point
for metric, value in metrics.items():
plt.text(metric, value, f'{value:.2f}', ha='center', va='bottom', fontsize=12, color='black')

# Set titles and labels
plt.title('Model Performance Metrics')
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.ylim(0, max(metrics.values()) + 10)  # Adjust y-axis limit for better visibility

# Show the plot
plt.grid(True)
plt.show()

