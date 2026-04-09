import pandas as pd
import shap
import dice_ml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('final_data.csv')
X = df.drop(['default.payment.next.month', 'ID'], axis=1)
y = df['default.payment.next.month']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(f"Model is trained, Accuracy: {model.score(X_test, y_test):.2f}")

explainer_shap = shap.TreeExplainer(model)
X_test_sample = X_test.iloc[:100] 
shap_values = explainer_shap.shap_values(X_test_sample)

print("\nSHAP Results")
if isinstance(shap_values, list):
    print("Top feature contributions for the first person:", shap_values[1][0])
    shap_plot_data = shap_values[1]
else:
    print("Top feature contributions for the first person:", shap_values[0])
    shap_plot_data = shap_values

print("\n5. Starting DiCE (this is the slow part)...")
d = dice_ml.Data(dataframe=df.drop('ID', axis=1),
                 continuous_features=['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'PAY_AMT1'], 
                 outcome_name='default.payment.next.month')
m = dice_ml.Model(model=model, backend='sklearn')
exp = dice_ml.Dice(d, m, method='random')

rejected_person = X_test[y_test == 1].iloc[0:1]
dice_exp = exp.generate_counterfactuals(rejected_person, total_CFs=3, desired_class=0)

print("\nDiCE Table Results")
print(dice_exp.cf_examples_list[0].final_cfs_df)
if isinstance(shap_values, list):
    shap.summary_plot(shap_values[1], X_test_sample, max_display=10)
else:
    shap.summary_plot(shap_values, X_test_sample, max_display=10)
plt.show()

