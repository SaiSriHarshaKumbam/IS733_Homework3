#!/usr/bin/env python
# coding: utf-8

# # Homework-3

# In[1]:


import pandas as pd
import numpy as np
filepath="C:/Users/harsha/Desktop/DataMining_733/Homework3/SP500_close_price_no_missing.csv"
df_sp500_close_no_missing=pd.read_csv(filepath)

shape=df_sp500_close_no_missing.shape
print("Shape of the dataset: ",shape)


# In[2]:


df_sp500_close_no_missing.head()


# In[3]:


df_sp500_close_no_missing.info()


# In[4]:


df_sp500_close_no_missing.dtypes


# In[5]:


#In the above lines of code date is being considered as an object datatype so I am converting it into date datatype to avaoid confusion in the further implementation of the code


# In[6]:


ddf_sp500_close_no_missing= df_sp500_close_no_missing.convert_dtypes()
df_sp500_close_no_missing['date']= pd.to_datetime(df_sp500_close_no_missing['date'])
df_sp500_close_no_missing.reset_index(inplace=True)
df_sp500_close_no_missing.dtypes


# In[7]:


#Now reading the rest of the stocks which were not listed because if the error or not getting listed as of Jan,1 2011 


# In[13]:


import pandas as pd

filepath_ticker = 'C:/Users/harsha/Desktop/DataMining_733/Homework3/SP500_ticker.csv'
df_sp500_ticker = pd.read_csv(filepath_ticker, encoding='latin1')  # or encoding='ISO-8859-1'
shape_ticker = df_sp500_ticker.shape
print("Shape of the dataset: ", shape_ticker)


# In[14]:


df_sp500_ticker.head()


# In[15]:


df_sp500_ticker.info()


# # Problem-1

# # a) Fit a PCA model to log returns

# # 1.	Derive log returns from the raw stock price dataset

# In[17]:



# Extracting numeric columns (excluding 'date')
numeric_columns = df_sp500_close_no_missing.drop('date', axis=1)

# Calculate log returns
log_returns = np.log(numeric_columns / numeric_columns.shift(1))

# Replace any potential inf or -inf values with NaN
log_returns.replace([np.inf, -np.inf], np.nan, inplace=True)

# Display the first few rows of log returns
print(log_returns.head())


# # 2. Plot a scree plot which shows the distribution of variance contained in subsequent principal components sorted by their eigenvalues.

# In[18]:


import matplotlib.pyplot as plt

# Assuming principal_components is the result of our PCA analysis

# Fit PCA model
from sklearn.decomposition import PCA

pca = PCA()
principal_components = pca.fit_transform(log_returns.dropna())

# Calculate the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Calculate cumulative explained variance
cumulative_explained_variance = explained_variance_ratio.cumsum()

# Plot the scree plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='-', color='b')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.show()


# # 3. Create a second plot showing cumulative variance retained if top N components are kept after dimensionality reduction (i.e. the horizontal axis will show the number of components kept, the vertical axis will show the cumulative percentage of variance retained).

# In[19]:


import matplotlib.pyplot as plt


# Assuming principal_components is the result of our PCA analysis

# Fit PCA model
from sklearn.decomposition import PCA

pca = PCA()
principal_components = pca.fit_transform(log_returns.dropna())

# Calculate the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Calculate cumulative explained variance
cumulative_explained_variance = explained_variance_ratio.cumsum()

# Plot the cumulative variance retained
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='-', color='r')
plt.title('Cumulative Variance Retained')
plt.xlabel('Number of Components Kept')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.show()


# # 4. How many principal components must be retained in order to capture at least 80% of the total variance in data?

# In[20]:


# Assuming cumulative_explained_variance is the cumulative explained variance obtained from your PCA analysis

# Set the threshold for the minimum variance retained (e.g., 80%)
variance_threshold = 0.8

# Find the number of components needed to reach or exceed the threshold
num_components_needed = np.argmax(cumulative_explained_variance >= variance_threshold) + 1

print(f'Number of components needed to retain at least {variance_threshold * 100}% of variance: {num_components_needed}')


# # b)	Analysis of principal components and weights 

# # 1.	Compute and plot the time series of the 1st principal component and observe temporal patterns. Identify the date with the lowest value for this component and conduct a quick research on the Internet to see if you can identify event(s) that might explain the observed behavior. 

# In[41]:


import matplotlib.pyplot as plt




# In[35]:


#cheking missing values
print(log_returns.isnull().sum())


# In[36]:


#checking for infinity or large values
print(np.isfinite(log_returns).all())
print((log_returns < np.finfo(np.float64).max).all())


# In[37]:


#removing missing or null values
log_returns = log_returns.dropna()  # Remove rows with missing values


# In[42]:


# Assuming principal_components is your array of principal components
# Extract the 1st principal component
pc1 = principal_components[:, 0]

# Check the dimensions and trim if necessary
df_sp500_close_no_missing['date'] = pd.to_datetime(df_sp500_close_no_missing['date'])
pc1_dates = df_sp500_close_no_missing['date'][:len(pc1)]

# Plot the time series of the 1st principal component
plt.figure(figsize=(12, 6))
plt.plot(pc1_dates, pc1)
plt.title('Time Series of 1st Principal Component')
plt.xlabel('Date')
plt.ylabel('Principal Component Value')
plt.show()


# In[40]:


# Identify the date with the lowest value
min_pc1_date = pc1_dates[np.argmin(pc1)]
print(f'Date with the lowest value for the 1st principal component: {min_pc1_date}')


# # 2.	Extract the weights from the PCA model for 1st and 2nd principal components. 

# In[43]:



# Extract weights for the 1st and 2nd principal components
weights_1st_pc = pca.components_[0]
weights_2nd_pc = pca.components_[1]

# Display the extracted weights
print("Weights for the 1st Principal Component:")
print(weights_1st_pc)

print("\nWeights for the 2nd Principal Component:")
print(weights_2nd_pc)


# In[48]:


# Assuming pca is your fitted PCA model
# Extract weights for the 1st and 2nd principal components
weights_1st_pc = pca.components_[0]
weights_2nd_pc = pca.components_[1]

# Display the extracted weights
print("Weights for the 1st Principal Component:")
print(weights_1st_pc)


# In[46]:



print("\nWeights for the 2nd Principal Component:")
print(weights_2nd_pc)


# # 3. Create a plot to show weights of the 1st principal component grouped by the industry sector (for example, you may draw a bar plot of mean weight per sector). Observe the distribution of weights (magnitudes, signs). Based on your observation, what kind of information do you think the 1st principal component might have captured?

# In[61]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Assuming df_sp500_ticker contains the ticker information, including the 'sector' column
# Assuming pca is your fitted PCA model

# Extract weights for the 1st principal component
weights_1st_pc = pca.components_[0]

# Create a DataFrame with tickers and corresponding weights for the 1st principal component
tickers = df_sp500_ticker['ticker'].values

# Ensure lengths match or adjust to the minimum length
min_length = min(len(tickers), len(weights_1st_pc))
df_weights_1st_pc = pd.DataFrame({'Ticker': tickers[:min_length], 'Weight_1st_PC': weights_1st_pc[:min_length]})

# Merge with sector information
df_weights_sector = pd.merge(df_weights_1st_pc, df_sp500_ticker[['ticker', 'sector']], left_on='Ticker', right_on='ticker', how='left')

# Group by sector and calculate the mean weight for the 1st principal component
mean_weights_by_sector = df_weights_sector.groupby('sector')['Weight_1st_PC'].mean()

# Plot bar plot
#plt.figure(figsize=(12, 6))
#mean_weights_by_sector.plot(kind='bar', color=np.where(mean_weights_by_sector > 0, 'b', 'r'))
#plt.title('Mean Weights of the 1st Principal Component by Sector')
#plt.xlabel('Industry Sector')
#plt.ylabel('Mean Weight')
#plt.show()


# In[62]:


import seaborn as sns
# Ensure the lengths match by considering only the relevant rows in df_sp500_ticker
df_weights_pc1 = pd.DataFrame({'Sector': df_sp500_ticker['sector'].iloc[:len(weights_1st_pc)], 'Weight_PC1': weights_1st_pc})

# Plot weights of the 1st principal component grouped by industry sector using seaborn
plt.figure(figsize=(12, 6))
sns.barplot(x='Sector', y='Weight_PC1', data=df_weights_pc1, ci=None, palette='viridis')
plt.title('Weights of 1st Principal Component by Industry Sector')
plt.xlabel('Industry Sector')
plt.ylabel('Mean Weight')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.show()


# # 4. Make a similar plot for the 2nd principal component.  What kind of information do you think does this component reveal? (Hint: look at the signs and magnitudes.)
# 

# In[70]:


df_sp500_ticker.info()


# In[74]:


# Assuming df_sp500_ticker contains the ticker information, including the 'sector' column
# Assuming pca is your fitted PCA model

# Extract weights for the 2nd principal component
weights_2nd_pc = pca.components_[1]

# Ensure lengths match or adjust to the minimum length
min_length = min(len(tickers), len(weights_2nd_pc))
df_weights_2nd_pc = pd.DataFrame({'Ticker': tickers[:min_length], 'Weight_2nd_PC': weights_2nd_pc[:min_length]})

# Merge with sector information
df_weights_sector_2nd_pc = pd.merge(df_weights_2nd_pc, df_sp500_ticker[['ticker', 'sector']], left_on='Ticker', right_on='ticker', how='left')

# Group by sector and calculate the mean weight for the 2nd principal component
mean_weights_by_sector_2nd_pc = df_weights_sector_2nd_pc.groupby('sector')['Weight_2nd_PC'].mean()

# Plot weights of the 2nd principal component grouped by industry sector using seaborn
plt.figure(figsize=(12, 8))
sns.barplot(x='sector', y='Weight_2nd_PC', data=df_weights_sector_2nd_pc, ci=None, palette='viridis')
plt.title('Weights of 2nd Principal Component by Industry Sector')
plt.xlabel('Industry Sector')
plt.ylabel('Mean Weight')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.show()


# # 5. Suppose we wanted to construct a new stock index using one principal component to track the overall market tendencies. Which of the two components would you prefer to use for this purpose, the 1st or the 2nd? Why?

# -> In this the 1st Principal Component represents the dominant trend in stock returns, it explains the maximum variance among all the components which are suitable for reflecting market wide components and trends

# -> Coming to the 2nd Principal component, capturing independent variance adds the diversification by capturing sector specific movements. This also represents additional, uncorrelated information

# -> The 1st principal compoment if preferred for closely tracking overall market tendencies, whereas the 2nd principal component is considered for diversification and capturing independent movements. 

# In[ ]:


#Bonus


# In[76]:


pip install yfinance


# In[77]:


import yfinance as yf
import pandas as pd

# Define the list of tickers you're interested in
tickers = ['AAPL', 'GOOGL', 'MSFT']  # Add more if needed

# Fetch historical stock prices
df_stock_prices = yf.download(tickers, start='2022-01-01', end='2023-01-01')['Adj Close']

# Print the first few rows of the data
print(df_stock_prices.head())

# Now you can proceed with the PCA analysis using the fetched data


# In[82]:


# Define the number of principal components you want to retain
n_components = 2  # Adjust this value as needed

# Assuming df_stock_prices contains the stock prices
log_returns = np.log(df_stock_prices / df_stock_prices.shift(1)).dropna()
standardized_returns = (log_returns - log_returns.mean()) / log_returns.std()
from sklearn.decomposition import PCA

# Instantiate PCA with the specified number of components
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(standardized_returns)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance Ratio')
plt.title('Cumulative Variance Explained by Principal Components')
plt.show()


# # Problem -2

# In[ ]:





# In[84]:


#Reading the bmi.csv file an dloading the dataset
filepath_bmi="C:/Users/harsha/Desktop/DataMining_733/Homework3/BMI.csv"
df_bmi=pd.read_csv(filepath_bmi)
#checking the shape of the dataset
shape=df_bmi.shape
print("Shape of the dataset: ",shape)


# In[85]:


df_bmi.head()


# In[86]:


# Display information about the dataset
print(df_bmi.info())


# In[87]:


# Check for missing values
print(df_bmi.isnull().sum())


# In[88]:


# Summary statistics
print(df_bmi.describe())


# In[90]:


# Assuming all columns except 'fatpctg' are features
X = df_bmi.drop('fatpctg', axis=1)

# Target variable
y = df_bmi['fatpctg']


# In[91]:


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[93]:


from sklearn.linear_model import LinearRegression

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)


# In[94]:


from sklearn.metrics import mean_squared_error, r2_score

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


# In[95]:


pip install mlxtend


# In[96]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


# Split the data into features (X) and target variable (y)
X = df_bmi.drop("fatpctg", axis=1)
y = df_bmi["fatpctg"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Forward stepwise feature selection
sfs_forward = SFS(model, k_features='best', forward=True, scoring='neg_mean_squared_error', cv=5)
sfs_forward.fit(X_train, y_train)

# Backward stepwise feature selection
sfs_backward = SFS(model, k_features='best', forward=False, scoring='neg_mean_squared_error', cv=5)
sfs_backward.fit(X_train, y_train)

# Print selected features for forward and backward stepwise selection
print("Features selected (Forward):", sfs_forward.k_feature_names_)
print("Features selected (Backward):", sfs_backward.k_feature_names_)


# In[100]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



# Split the data into features (X) and target variable (y)
X = df_bmi.drop("fatpctg", axis=1)
y = df_bmi["fatpctg"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Function to perform forward stepwise regression
def forward_stepwise_selection(X, y):
    selected_features = []
    remaining_features = list(X.columns)

    while remaining_features:
        mse_scores = []

        for feature in remaining_features:
            candidate_features = selected_features + [feature]
            X_subset = X[candidate_features]
            X_train, X_val, y_train, y_val = train_test_split(X_subset, y, test_size=0.2, random_state=42)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            mse_scores.append(mse)

        best_feature_index = np.argmin(mse_scores)
        best_feature = remaining_features[best_feature_index]
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)

    return selected_features

# Function to perform backward stepwise regression
def backward_stepwise_selection(X, y):
    selected_features = list(X.columns)

    while len(selected_features) > 1:  # Ensure at least one feature is selected
        mse_scores = []

        for feature in selected_features:
            candidate_features = selected_features.copy()
            candidate_features.remove(feature)
            X_subset = X[candidate_features]

            # Ensure the data is in the correct format (numpy array)
            if isinstance(X_subset, pd.DataFrame):
                X_subset = X_subset.values
            if isinstance(y, pd.Series):
                y = y.values

            X_train, X_val, y_train, y_val = train_test_split(X_subset, y, test_size=0.2, random_state=42)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            mse_scores.append(mse)

        worst_feature_index = np.argmax(mse_scores)
        worst_feature = selected_features[worst_feature_index]
        selected_features.remove(worst_feature)

    return selected_features

# Perform forward stepwise selection
forward_selected_features = forward_stepwise_selection(X_train, y_train)
print("Selected features (Forward):", forward_selected_features)

# Perform backward stepwise selection
backward_selected_features = backward_stepwise_selection(X_train, y_train)
print("Selected features (Backward):", backward_selected_features)


# In[103]:


pip install --upgrade scikit-learn


# In[108]:


pip install scikit-learn==0.24.2


# In[109]:




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import seaborn as sns

# Load the data
data = pd.read_csv('C:/Users/harsha/Desktop/DataMining_733/Homework3/BMI.csv')

# Separate features (X) and target variable (y)
X = data.drop('fatpctg', axis=1)
y = data['fatpctg']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Method 1: Feature Importance from Tree-based models (e.g., RandomForest)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Display feature importance
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importance.sort_values(ascending=False).plot(kind='barh')
plt.title("Feature Importance from RandomForest")
plt.show()

# Select features based on importance
sfm = SelectFromModel(rf_model, threshold='median')
sfm.fit(X_train, y_train)
selected_features = X_train.columns[sfm.get_support()]

# Method 2: Correlation-based feature selection
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# Select features based on correlation threshold (e.g., 0.2)
correlation_threshold = 0.2
selected_features_corr = [col for col in correlation_matrix.columns if abs(correlation_matrix['fatpctg'][col]) > correlation_threshold]

# You can now use the selected features for further modeling

# Example: Use the selected features for modeling
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Continue with your preferred modeling approach (e.g., linear regression, support vector machines, etc.)
# ...

# Evaluate your model and compare the results
# ...


# In[110]:


from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Create an SVM regressor
svm_regressor = SVR()

# Fit the SVM model using the selected features from Random Forest
svm_regressor.fit(X_train_selected, y_train)

# Make predictions on the test set
y_pred = svm_regressor.predict(X_test_selected)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')


# In[111]:


# a) Wrapper method: 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Initialize an empty set for selected features
selected_features_forward = []

# Create a linear regression model
model_forward = LinearRegression()

# Forward stepwise regression
for feature in X_train.columns:
    features_to_try = selected_features_forward + [feature]
    model_forward.fit(X_train[features_to_try], y_train)
    y_pred_forward = model_forward.predict(X_test[features_to_try])
    mse_forward = mean_squared_error(y_test, y_pred_forward)
    
    if not selected_features_forward or mse_forward < mse_forward_best:
        mse_forward_best = mse_forward
        selected_features_forward = features_to_try

print(f"Selected features (Forward Stepwise Regression): {selected_features_forward}")
print(f"Mean Squared Error (Forward Stepwise Regression): {mse_forward_best}")


# In[112]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create a linear regression model
model_backward = LinearRegression()

# Backward stepwise regression
selected_features_backward = list(X_train.columns)

while len(selected_features_backward) > 0:
    model_backward.fit(X_train[selected_features_backward], y_train)
    y_pred_backward = model_backward.predict(X_test[selected_features_backward])
    mse_backward = mean_squared_error(y_test, y_pred_backward)
    
    mse_without_feature = []

    for feature in selected_features_backward:
        features_to_try = [f for f in selected_features_backward if f != feature]
        model_backward.fit(X_train[features_to_try], y_train)
        y_pred_without_feature = model_backward.predict(X_test[features_to_try])
        mse_without_feature.append(mean_squared_error(y_test, y_pred_without_feature))

    mse_without_feature_min = min(mse_without_feature)

    if mse_without_feature_min < mse_backward:
        selected_features_backward.pop(mse_without_feature.index(mse_without_feature_min))
    else:
        break

print(f"Selected features (Backward Stepwise Regression): {selected_features_backward}")
print(f"Mean Squared Error (Backward Stepwise Regression): {mse_without_feature_min}")


# In[113]:


# b). Filter Method
import pandas as pd


# Separate features (X) and target variable (y)
X = data.drop('fatpctg', axis=1)
y = data['fatpctg']

# Calculate correlation between input variables and output variable
correlation_with_output = X.apply(lambda x: x.corr(y))

# Sort features based on correlation
sorted_features = correlation_with_output.abs().sort_values(ascending=False)

# Output the ranking
print("Feature Ranking based on Correlation with Output:")
for feature, correlation in sorted_features.items():
    print(f"{feature}: {correlation}")


# # Embedded method
# 1. Lasso regression

# In[114]:


from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a Lasso regression model
lasso_model = Lasso(alpha=0.1)  # You can adjust the alpha parameter

# Fit the model
lasso_model.fit(X_train_scaled, y_train)

# Get selected features based on non-zero coefficients
selected_features_lasso = X.columns[lasso_model.coef_ != 0]

# Print selected features
print("Selected features (Lasso Regression):", selected_features_lasso)


# 2. Random Forest

# In[116]:


from sklearn.ensemble import RandomForestRegressor

# Create a Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model
rf_model.fit(X_train, y_train)

# Get feature importance scores
feature_importance_rf = pd.Series(rf_model.feature_importances_, index=X.columns)

# Sort features based on importance
selected_features_rf = feature_importance_rf.sort_values(ascending=False).index

# Print selected features
print("Selected features (Random Forest):", selected_features_rf)


# In[ ]:




