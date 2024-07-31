# Linear_regression
It is like you are working on multiple data analysis tasks, including visualizing and analyzing datasets, performing linear regression, and evaluating models. Let's break it down and address each part:

### 1. Data Visualization

#### Plotting Gender over Age
```python
import matplotlib.pyplot as plt

# Plotting Gender over Age
plt.scatter(df['Gender'], df['Age'])
plt.xlabel('Gender')
plt.ylabel('Age')
plt.title('Gender over Age')
plt.show()
```

#### Bar Plot of Customer ID over Size
```python
plt.bar(df['Customer ID'], df['Size'])
plt.xlabel('Customer ID')
plt.ylabel('Size')
plt.title('Customer ID over Size')
plt.show()
```

#### Plotting Category over Season
```python
plt.plot(df['Category'], df['Season'])
plt.xlabel('Category')
plt.ylabel('Season')
plt.title('Category over Season')
plt.show()
```

### 2. Linear Regression Analysis on Shopping Trends Dataset
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv(r"C:\Users\Yuktha\Desktop\shopping_trends_updated.csv")

# Feature selection
X = df[['Purchase Amount (USD)']]
Y = df['Previous Purchases']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35, random_state=46)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train linear regression model
regression = LinearRegression()
regression.fit(X_train, Y_train)

# Make predictions
Y_pred = regression.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, Y_pred)
adjusted_r2 = 1 - (1-r2)*(len(Y_test)-1)/(len(Y_test)-X_test.shape[1]-1)

# Print evaluation metrics
print("Coefficient (slope):", regression.coef_)
print("Intercept:", regression.intercept_)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)
print("Adjusted R-squared:", adjusted_r2)

# Plot results
plt.scatter(X_test, Y_test, color='black')
plt.plot(X_test, Y_pred, color='blue', linewidth=3)
plt.xlabel('Purchase Amount (USD)')
plt.ylabel('Previous Purchases')
plt.title('Linear Regression: Purchase Amount vs Previous Purchases')
plt.show()
```

### 3. Visualizing Height and Weight Dataset

#### Scatter Plot of Weight vs Height
```python
df = pd.read_csv(r"C:\Users\Yuktha\Desktop\height-weight.csv")

# Scatter plot
plt.scatter(df['Weight(Pounds)'], df['Height(Inches)'])
plt.xlabel('Weight (Pounds)')
plt.ylabel('Height (Inches)')
plt.title('Weight vs Height')
plt.show()
```

#### Pairplot
```python
import seaborn as sns

sns.pairplot(df)
plt.show()
```

### 4. Linear Regression Analysis on Height-Weight Dataset
```python
X = df[['Weight(Pounds)']]
Y = df['Height(Inches)']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35, random_state=46)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train linear regression model
regression = LinearRegression()
regression.fit(X_train, Y_train)

# Make predictions
Y_pred = regression.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, Y_pred)
adjusted_r2 = 1 - (1-r2)*(len(Y_test)-1)/(len(Y_test)-X_test.shape[1]-1)

# Print evaluation metrics
print("Coefficient (slope):", regression.coef_)
print("Intercept:", regression.intercept_)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)
print("Adjusted R-squared:", adjusted_r2)

# Plot results
plt.scatter(X_test, Y_test, color='black')
plt.plot(X_test, Y_pred, color='blue', linewidth=3)
plt.xlabel('Weight (Pounds)')
plt.ylabel('Height (Inches)')
plt.title('Linear Regression: Weight vs Height')
plt.show()
```

### 5. Ordinary Least Squares (OLS) Regression using Statsmodels
```python
import statsmodels.api as sm

# Add a constant term for the intercept
X_train_sm = sm.add_constant(X_train)

# Fit OLS model
model = sm.OLS(Y_train, X_train_sm).fit()

# Print model summary
print(model.summary())
```

Ensure the data file paths are correct when loading datasets. These scripts cover visualizations, linear regression analysis, and evaluation of the models. Adjust any plot parameters as needed to improve visualization clarity.
