import pandas as pd

# Create first CSV
basic_results = pd.DataFrame({
    'Classifier': ['Logistic Regression', 'Multinomial Naive Bayes', 'Linear SVC', 
                  'Random Forest', 'Gradient Boosting', 'XGBoost'],
    'Accuracy': [0.8833, 0.8525, 0.8801, 0.8487, 0.8088, 0.8575]
})
basic_results.to_csv('classifier_basic_results.csv', index=False)

# Create second CSV
detailed_models = pd.DataFrame({
    'Model': ['SVM', 'DecisionTree', 'NaiveBayes', 'RandomForest', 'Optimized', 'Stacking'],
    'Accuracy': [0.82, 0.63, 0.62, 0.81, 0.83, 0.81],
    'Positive_Precision': [0.80, 0.62, 0.61, 0.80, 0.79, 0.80],
    'Positive_Recall': [0.86, 0.64, 0.66, 0.83, 0.88, 0.82],
    'Negative_Precision': [0.85, 0.63, 0.64, 0.82, 0.87, 0.82],
    'Negative_Recall': [0.78, 0.62, 0.59, 0.79, 0.77, 0.80]
})
detailed_models.to_csv('classifier_detailed_results.csv', index=False)