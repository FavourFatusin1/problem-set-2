'''
PART 3: Logistic Regression
- Read in `df_arrests`
- Use train_test_split to create two dataframes from `df_arrests`, the first is called `df_arrests_train` and the second is called `df_arrests_test`. Set test_size to 0.3, shuffle to be True. Stratify by the outcome  
- Create a list called `features` which contains our two feature names: pred_universe, num_fel_arrests_last_year
- Create a parameter grid called `param_grid` containing three values for the C hyperparameter. (Note C has to be greater than zero) 
- Initialize the Logistic Regression model with a variable called `lr_model` 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv` 
- Run the model 
- What was the optimal value for C? Did it have the most or least regularization? Or in the middle? Print these questions and your answers. 
- Now predict for the test set. Name this column `pred_lr`
- Return dataframe(s) for use in main.py for PART 4 and PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 4 and PART 5 in main.py
'''

# Import any further packages you may need for PART 3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.linear_model import LogisticRegression as lr


# Your code here
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Step 1: Read in `df_arrests`
df_arrests = pd.read_csv('data/df_arrests.csv')

# Step 2: Split data into training and test sets
df_arrests_train, df_arrests_test = train_test_split(
    df_arrests, 
    test_size=0.3, 
    shuffle=True, 
    stratify=df_arrests['y'],  # Stratify by the outcome
    random_state=42  # For reproducibility
)

# Step 3: Define features and target variable
features = ['num_fel_arrests_last_year', 'current_charge_felony']
X_train = df_arrests_train[features]
y_train = df_arrests_train['y']
X_test = df_arrests_test[features]
y_test = df_arrests_test['y']

# Step 4: Create parameter grid for C hyperparameter
param_grid = {'C': [0.01, 0.1, 1]}

# Step 5: Initialize Logistic Regression model
lr_model = LogisticRegression()

# Step 6: Initialize GridSearchCV
gs_cv = GridSearchCV(
    estimator=lr_model, 
    param_grid=param_grid, 
    cv=5,  # 5-fold cross-validation
    scoring='accuracy'  # Metric to optimize
)

# Step 7: Fit the model
gs_cv.fit(X_train, y_train)

# Step 8: Optimal value for C
best_C = gs_cv.best_params_['C']
print(f"The optimal value for C is: {best_C}")

# Interpret regularization strength
if best_C < 0.1:
    reg_strength = "most regularization"
elif best_C == 0.1:
    reg_strength = "in the middle"
else:
    reg_strength = "least regularization"
print(f"Did it have the most or least regularization? Or in the middle? {reg_strength}")

# Step 9: Predict for the test set
df_arrests_test['pred_lr'] = gs_cv.predict(X_test)

# Step 10: Save results for use in main.py
df_arrests_test.to_csv('data/df_arrests_test_with_predictions.csv', index=False)


