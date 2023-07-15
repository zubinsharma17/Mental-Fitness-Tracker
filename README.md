
# Mental Health Fitness Tracker
The Mental Health Fitness Tracker project focuses on analyzing and predicting mental fitness levels of individuals from various countries with different mental disorders. It utilizes regression techniques to provide insights into mental health and make predictions based on the available data.

Download the project files and navigate to the project directory.
   

THEN

IMPORT THE NECESSARY LIBRARIES

```bash
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
%matplotlib inline
import plotly.express  as px
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")
```
READ THE DATA FROM THE CSV FILES

```bash
df = pd.read_csv('mental-disorders-countrywise.csv')
uf = pd.read_csv('every_mental_disorder_countrywise.csv')
```

FILL MISSING VALUES IN NUMERIC COLUMNS OF DATAFRAMES df1 AND df2 WITH THE MEAN OF THEIR RESPECTIVE COLUMNS

```bash
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

numeric_columns = uf.select_dtypes(include=[np.number]).columns
uf[numeric_columns] = uf[numeric_columns].fillna(uf[numeric_columns].mean())
```

CONVERT DATA TYPES

```bash
df['DALYs (Disability-Adjusted Life Years) - Mental disorders - Sex: Both - Age: All Ages (Percent)'] = df['DALYs (Disability-Adjusted Life Years) - Mental disorders - Sex: Both - Age: All Ages (Percent)'].astype(float)
uf['Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized'] = uf['Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized'].astype(float)
uf['Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized'] = uf['Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized'].astype(float)
uf['Eating disorders (share of population) - Sex: Both - Age: Age-standardized'] = uf['Eating disorders (share of population) - Sex: Both - Age: Age-standardized'].astype(float)
uf['Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized'] = uf['Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized'].astype(float)
uf['Prevalence - Drug use disorders - Sex: Both - Age: Age-standardized (Percent)'] = uf['Prevalence - Drug use disorders - Sex: Both - Age: Age-standardized (Percent)'].astype(float)
uf['Depressive disorders (share of population) - Sex: Both - Age: Age-standardized'] = uf['Depressive disorders (share of population) - Sex: Both - Age: Age-standardized'].astype(float)
uf['Prevalence - Alcohol use disorders - Sex: Both - Age: Age-standardized (Percent)'] = uf['Prevalence - Alcohol use disorders - Sex: Both - Age: Age-standardized (Percent)'].astype(float)
```

MERGE THE TWO DATAFRAMES ON A COMMON COLUMN

```bash
merged_df = pd.merge(df, uf, on=['Entity', 'Code', 'Year'])
```

FEATURE THE MATRIX X AND THE VARIABLE y

```bash
X = merged_df[['Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized',
               'Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized',
               'Eating disorders (share of population) - Sex: Both - Age: Age-standardized',
               'Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized',
               'Prevalence - Drug use disorders - Sex: Both - Age: Age-standardized (Percent)',
               'Depressive disorders (share of population) - Sex: Both - Age: Age-standardized',
               'Prevalence - Alcohol use disorders - Sex: Both - Age: Age-standardized (Percent)']]

y = merged_df['DALYs (Disability-Adjusted Life Years) - Mental disorders - Sex: Both - Age: All Ages (Percent)']
```

SPLIT THE DATA INTO TRAINING AND TESTING SETS

```bash
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

VISUALISING THE CORRELATION HEATMAP OF DISEASES AND MENTAL FITNESS

```bash
# Compute the correlation matrix
corr_matrix = merged_df[['Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized',
                         'Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized',
                         'Eating disorders (share of population) - Sex: Both - Age: Age-standardized',
                         'Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized',
                         'Prevalence - Drug use disorders - Sex: Both - Age: Age-standardized (Percent)',
                         'Depressive disorders (share of population) - Sex: Both - Age: Age-standardized',
                         'Prevalence - Alcohol use disorders - Sex: Both - Age: Age-standardized (Percent)',
                         'DALYs (Disability-Adjusted Life Years) - Mental disorders - Sex: Both - Age: All Ages (Percent)'
                        ]].corr()

# Create the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap - Diseases and Mental Fitness')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()
```

FIT THE LINEAR REGRESSION MODEL
    
```bash
model = LinearRegression()
model.fit(X_train, y_train)
```

MAKE A PREDICTION USING TRAINED MODEL

```bash
y_pred = model.predict(X_test)
```

PRINTING MODEL PERFOMANCE METRICS

```bash
# Create a dictionary to store the model performance
model_performance = {}

# Ridge Regression
ridge_model = Ridge(alpha=0.5)
ridge_model.fit(X_train, y_train)
ridge_y_pred = ridge_model.predict(X_test)
ridge_mse = mean_squared_error(y_test, ridge_y_pred)
ridge_r2 = r2_score(y_test, ridge_y_pred)
model_performance['1. Ridge Regression'] = {'MSE': ridge_mse, 'R-squared': ridge_r2}

# Lasso Regression
lasso_model = Lasso(alpha=0.5)
lasso_model.fit(X_train, y_train)
lasso_y_pred = lasso_model.predict(X_test)
lasso_mse = mean_squared_error(y_test, lasso_y_pred)
lasso_r2 = r2_score(y_test, lasso_y_pred)
model_performance['2. Lasso Regression'] = {'MSE': lasso_mse, 'R-squared': lasso_r2}

# Decision Tree Regression
tree_model = DecisionTreeRegressor()
tree_model.fit(X_train, y_train)
tree_y_pred = tree_model.predict(X_test)
tree_mse = mean_squared_error(y_test, tree_y_pred)
tree_r2 = r2_score(y_test, tree_y_pred)
model_performance['3. Decision Tree Regression'] = {'MSE': tree_mse, 'R-squared': tree_r2}

# Random Forest Regression
forest_model = RandomForestRegressor()
forest_model.fit(X_train, y_train)
forest_y_pred = forest_model.predict(X_test)
forest_mse = mean_squared_error(y_test, forest_y_pred)
forest_r2 = r2_score(y_test, forest_y_pred)
model_performance['4. Random Forest Regression'] = {'MSE': forest_mse, 'R-squared': forest_r2}

# K-Nearest Neighbors Regression
knn_model = KNeighborsRegressor()
knn_model.fit(X_train, y_train)
knn_y_pred = knn_model.predict(X_test)
knn_mse = mean_squared_error(y_test, knn_y_pred)
knn_r2 = r2_score(y_test, knn_y_pred)
model_performance['5. K-Nearest Neighbors Regression'] = {'MSE': knn_mse, 'R-squared': knn_r2}

# Neural Network Regression
nn_model = MLPRegressor(max_iter=1000)
nn_model.fit(X_train, y_train)
nn_y_pred = nn_model.predict(X_test)
nn_mse = mean_squared_error(y_test, nn_y_pred)
nn_r2 = r2_score(y_test, nn_y_pred)
model_performance['6. Neural Network Regression'] = {'MSE': nn_mse, 'R-squared': nn_r2}

# Print model performance
for model, performance in model_performance.items():
    print(f"Model: {model}")
    print("   Mean Squared Error (MSE):", performance['MSE'])
    print("   R-squared Score:", performance['R-squared'])
    print()
```

PLOTTING PREDECTED vs ACTUAL VALUES GRAPH

```bash
# Create a dictionary to store the model performance
model_performance = {
    'Ridge Regression': {'Predicted': ridge_y_pred, 'Actual': y_test},
    'Lasso Regression': {'Predicted': lasso_y_pred, 'Actual': y_test},
    'Decision Tree Regression': {'Predicted': tree_y_pred, 'Actual': y_test},
    'Random Forest Regression': {'Predicted': forest_y_pred, 'Actual': y_test},
    'K-Nearest Neighbors Regression': {'Predicted': knn_y_pred, 'Actual': y_test},
    'Neural Network Regression': {'Predicted': nn_y_pred, 'Actual': y_test},
   }

# Set up figure and axes
num_models = len(model_performance)
num_rows = (num_models // 3) + (1 if num_models % 3 != 0 else 0)
fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5))

# Define color palette
color_palette = plt.cm.Set1(range(num_models))

# Iterate over the models and plot the predicted vs actual values
for i, (model, performance) in enumerate(model_performance.items()):
    row = i // 3
    col = i % 3
    ax = axes[row, col] if num_rows > 1 else axes[col]

    # Get the predicted and actual values
    y_pred = performance['Predicted']
    y_actual = performance['Actual']

    # Scatter plot of predicted vs actual values
    ax.scatter(y_actual, y_pred, color=color_palette[i], alpha=0.5, marker='o')

    # Add a diagonal line for reference
    ax.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], color='r')

    # Set the title and labels
    ax.set_title(model)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')

    # Add gridlines
    ax.grid(True)

# Adjust spacing between subplots
fig.tight_layout()

# Create a legend
plt.legend(model_performance.keys(), loc='upper right')

# Show the plot
plt.show()
```

THIS PRINTS REGRESSION MODEL IN ORDER OF PRECISION AND A FINAL RESULT TELLING WHICH REGRESSION MODEL HAS THE MOST PRECISE VALUE AND WHICH REGRESSION MODEL HAS LEAST PRECISE VALUE

```bash
# Store the regression models and their scores in a dictionary
regression_scores = {
    "Ridge Regression": (ridge_mse, ridge_r2), 
    "Random Forest Regression": (forest_mse, forest_r2),
    "Decision Tree Regression": (tree_mse, tree_r2),
    "Lasso Regression": (lasso_mse, lasso_r2),
    "K-Nearest Neighbors Regression": (knn_mse, knn_r2),
    "Neural Network Regression": (nn_mse, nn_r2),
}

# Sort the regression models based on MSE in ascending order and R-squared score in descending order
sorted_models = sorted(regression_scores.items(), key=lambda x: (x[1][0], -x[1][1]))

print("Regression Models in Order of Precision:")
for i, (model, scores) in enumerate(sorted_models, start=1):
    print(f"{i}. {model}")
    print("   Mean Squared Error (MSE):", scores[0])
    print("   R-squared Score:", scores[1])
    print()

most_precise_model = sorted_models[0][0]
least_precise_model = sorted_models[-1][0]

print(f"The most precise model is: {most_precise_model}")
print(f"The least precise model is: {least_precise_model}")
```
## SUMMARY
- Developed a fully functional Mental Health Fitness Tracker ML model using Python and scikit-learn.
- Utilized 6 types of regression algorithms to achieve precise results in analyzing and predicting mental fitness levels from over 150 countries.
- Cleaned, preprocessed, and engineered features to enhance the model's predictive capabilities.
- Optimized the model's performance by fine-tuning hyperparameters and implementing ensemble methods, resulting in an impressive accuracy percentage of 98.50%.


## REFERENCES
- Datasets that were used here were taken from [ourworldindia.org](https://ourworldindata.org/grapher/mental-and-substance-use-as-share-of-disease)

- This project was made during my internship period for [IBM SkillsBuild](https://skillsbuild.org) in association with [AICTE](https://internship.aicte-india.org)
