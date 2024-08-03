'''
PART 4: Decision Trees
- Read in the dataframe(s) from PART 3
- Create a parameter grid called `param_grid_dt` containing three values for tree depth. (Note C has to be greater than zero) 
- Initialize the Decision Tree model. Assign this to a variable called `dt_model`. 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`. 
- Run the model 
- What was the optimal value for max_depth?  Did it have the most or least regularization? Or in the middle? 
- Now predict for the test set. Name this column `pred_dt` 
- Return dataframe(s) for use in main.py for PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 5 in main.py
'''

# Import any further packages you may need for PART 4
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.tree import DecisionTreeClassifier as DTC

from sklearn.tree import DecisionTreeClassifier

# Read in the dataframe(s) from PART 3
df_arrests = pd.read_csv('data/df_arrests.csv')  # Adjust the path if necessary

# Prepare the features and target variable
X = df_arrests[['num_fel_arrests_last_year', 'current_charge_felony']]
y = df_arrests['y']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y)

# Create a parameter grid for Decision Tree
param_grid_dt = {
    'max_depth': [3, 5, 7]  # Adjust the values as needed
}

# Initialize the Decision Tree model
dt_model = DecisionTreeClassifier()

# Initialize GridSearchCV
gs_cv_dt = GridSearchCV(estimator=dt_model, param_grid=param_grid_dt, cv=5, scoring='accuracy')

# Run the model
gs_cv_dt.fit(X_train, y_train)

# Optimal value for max_depth
best_max_depth = gs_cv_dt.best_params_['max_depth']
print(f"Optimal value for max_depth: {best_max_depth}")

# Predict for the test set
df_arrests_test = X_test.copy()
df_arrests_test['pred_dt'] = gs_cv_dt.predict(X_test)

# Print optimal max_depth and its regularization
if best_max_depth == min(param_grid_dt['max_depth']):
    print(f"The optimal max_depth value of {best_max_depth} has the most regularization.")
elif best_max_depth == max(param_grid_dt['max_depth']):
    print(f"The optimal max_depth value of {best_max_depth} has the least regularization.")
else:
    print(f"The optimal max_depth value of {best_max_depth} is in the middle of the regularization range.")

# Save DataFrame(s) for use in PART 5
df_arrests_test.to_csv('data/df_arrests_test_with_preds.csv', index=False)
