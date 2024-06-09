




import sys
sys.path.append('.')
import pandas as pd
from src.CMR1.config import conf_init
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix
)


config=conf_init()
models=config['models']
domains=config['domains']
input_path=config['CMR1_evaluated_data_dir']
#df=pd.read_csv("results/CMR2/evaluated_data/gpt-4-turbo_model_prediction.csv")
#df=pd.read_csv("results/CMR2/evaluated_data/gpt-3.5-turbo_model_prediction.csv")
#df=pd.read_csv("results/CMR2/evaluated_data/llama3-70b_model_prediction.csv")
#df=pd.read_csv("results/CMR2/evaluated_data/mixtral-8x22b-instruct_model_prediction.csv")

df=pd.read_csv("results/CMR1/evaluated_data/cos_sim_text-embedding-3-large.csv")
df['threshold']=df.apply(lambda row: row['text1_text2_cos_sim_text-embedding-3-large']>0.8,axis=1)

y_pred=df['threshold']

y_test=df["Same Causal Variable"]



accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred,zero_division=0)
recall = recall_score(y_test, y_pred,zero_division=0)
f1 = f1_score(y_test, y_pred,zero_division=0)
conf_matrix = confusion_matrix(y_test, y_pred)
kappa= cohen_kappa_score(y_test, y_pred)
print(f"precision_score: {precision:.2f}")
print(f"recall_score: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"cohen_kappa Score: {kappa:.2f}")

for model in models:

    print(f"{model} is performing in genal")
        

    df=pd.read_csv(input_path+f"{model}_model_prediction.csv")


    y_pred=df["Predicted Same Causal Variable"]

    y_test=df["Generated Same Causal Variable"]



    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred,zero_division=0)
    recall = recall_score(y_test, y_pred,zero_division=0)
    f1 = f1_score(y_test, y_pred,zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    kappa= cohen_kappa_score(y_test, y_pred)

    print(f"F1 Score: {f1:.2f}")
    print(f"cohen_kappa Score: {kappa:.2f}")


for model in models:


   

    df=pd.read_csv(input_path+f"{model}_model_prediction.csv")

    for domain in domains:

        print(f"{model} is performing in {domain}")
        
        tmp=df[df['Domain']==domain]

    

        y_pred=tmp["Predicted Same Causal Variable"]

        y_test=tmp["Generated Same Causal Variable"]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred,zero_division=0)
        recall = recall_score(y_test, y_pred,zero_division=0)
        f1 = f1_score(y_test, y_pred,zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)
        kappa= cohen_kappa_score(y_test, y_pred,)

        #print(f"Accuracy: {accuracy:.2f}")
        #print(f"Precision: {precision:.2f}")
        #print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print(f"cohen_kappa Score: {kappa:.2f}")


for model in models:


    df=pd.read_csv(input_path+f"{model}_model_prediction.csv")

    for generation_model in models:

        print(f"{model} is agreeing with {generation_model}")
        
        tmp=df[df['Data Generation Model']==generation_model]

    

        y_pred=tmp["Predicted Same Causal Variable"]

        y_test=tmp["Generated Same Causal Variable"]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred,zero_division=0)
        recall = recall_score(y_test, y_pred,zero_division=0)
        f1 = f1_score(y_test, y_pred,zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)
        kappa= cohen_kappa_score(y_test, y_pred,)

        #print(f"Accuracy: {accuracy:.2f}")
        #print(f"Precision: {precision:.2f}")
        #print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print(f"cohen_kappa Score: {kappa:.2f}")
