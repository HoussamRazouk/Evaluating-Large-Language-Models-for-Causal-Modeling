import pandas as pd
import numpy as np
import sys
sys.path.append('.')
import pandas as pd
from src.CMR2.config import conf_init


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


config=conf_init()
models=config['models']
output_path=config['CMR2_evaluated_data_dir']
domains=config['domains']
#df=pd.read_csv("results/CMR2/evaluated_data/gpt-4-turbo_model_prediction.csv")
#df=pd.read_csv("results/CMR2/evaluated_data/gpt-3.5-turbo_model_prediction.csv")
#df=pd.read_csv("results/CMR2/evaluated_data/llama3-70b_model_prediction.csv")
#df=pd.read_csv("results/CMR2/evaluated_data/mixtral-8x22b-instruct_model_prediction.csv")


for model in models:

    print(model)

    df=pd.read_csv(output_path+f"{model}_model_prediction.csv")

    y_pred=df["Predicted Interaction value"]

    y_test=df["Generated Interaction value"]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")



for model in models:


    print(model)

    df=pd.read_csv(output_path+f"{model}_model_prediction.csv")

    for domain in domains:

        print(domain)
        tmp=df[df['Domain']==domain]

    

        y_pred=tmp["Predicted Interaction value"]

        y_test=tmp["Generated Interaction value"]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        conf_matrix = confusion_matrix(y_test, y_pred)

        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")