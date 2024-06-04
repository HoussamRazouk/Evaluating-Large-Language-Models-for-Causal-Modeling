import pandas as pd


#df=pd.read_csv("results/evaluated_data/gpt-4-turbo_model_prediction.csv")
#df=pd.read_csv("results/evaluated_data/gpt-3.5-turbo_model_prediction.csv")
#df=pd.read_csv("results/evaluated_data/llama3-70b_model_prediction.csv")
df=pd.read_csv("results/evaluated_data/mixtral-8x22b-instruct_model_prediction.csv")



y_pred=df["Predicted Same Causal Variable"]

y_test=df["Generated Same Causal Variable"]


import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")