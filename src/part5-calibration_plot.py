'''
PART 5: Calibration-light
Use `calibration_plot` function to create a calibration curve for the logistic regression model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Use `calibration_plot` function to create a calibration curve for the decision tree model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Which model is more calibrated? Print this question and your answer. 

Extra Credit
Compute  PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
Compute  PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
Compute AUC for the logistic regression model
Compute AUC for the decision tree model
Do both metrics agree that one model is more accurate than the other? Print this question and your answer. 
'''
# Import any further packages you may need for PART 5
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Calibration plot function 
def calibration_plot(y_true, y_prob, n_bins=10):
    """
    Create a calibration plot with a 45-degree dashed line.

    Parameters:
        y_true (array-like): True binary labels (0 or 1).
        y_prob (array-like): Predicted probabilities for the positive class.
        n_bins (int): Number of bins to divide the data for calibration.

    Returns:
        None
    """
    # Calculate calibration values
    bin_means, prob_true = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    # Create the Seaborn plot
    sns.set(style="whitegrid")
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(prob_true, bin_means, marker='o', label="Model")
    
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend(loc="best")
    plt.show()

# Load data
df_arrests_test = pd.read_csv('data/df_arrests_test_with_predictions.csv')

# Extract true labels and predicted probabilities
y_true = df_arrests_test['y']
y_prob_lr = df_arrests_test['pred_lr']
y_prob_dt = df_arrests_test['pred_dt']

# Create calibration plots
print("Creating calibration plot for Logistic Regression...")
calibration_plot(y_true, y_prob_lr, n_bins=5)

print("Creating calibration plot for Decision Tree...")
calibration_plot(y_true, y_prob_dt, n_bins=5)

# Determine which model is more calibrated
# Calibration plots closer to the 45-degree line indicate better calibration
print("Which model is more calibrated? Based on visual inspection of calibration plots.")

# Extra Credit

# Compute PPV for top 50
top_50_lr = df_arrests_test.nlargest(50, 'pred_lr')
top_50_dt = df_arrests_test.nlargest(50, 'pred_dt')

# PPV calculations
ppv_lr = precision_score(top_50_lr['y'], top_50_lr['pred_lr'] >= 0.5)
ppv_dt = precision_score(top_50_dt['y'], top_50_dt['pred_dt'] >= 0.5)

print(f"PPV for Logistic Regression (top 50): {ppv_lr:.4f}")
print(f"PPV for Decision Tree (top 50): {ppv_dt:.4f}")

# Compute AUC
auc_lr = roc_auc_score(y_true, y_prob_lr)
auc_dt = roc_auc_score(y_true, y_prob_dt)

print(f"AUC for Logistic Regression: {auc_lr:.4f}")
print(f"AUC for Decision Tree: {auc_dt:.4f}")

# Do both metrics agree that one model is more accurate than the other?
better_model = "Logistic Regression" if auc_lr > auc_dt else "Decision Tree"
print(f"Do both metrics agree that one model is more accurate than the other? {better_model} is more accurate based on AUC.")
