import sys
sys.path.append('.')
import pandas as pd
from src.CMR2.config import conf_init
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score
)

def get_metrics(y_pred,y_test,results_file):

    accuracy = round(accuracy_score(y_test, y_pred),2)
    precision = round(precision_score(y_test, y_pred,zero_division=0),2)
    recall = round(recall_score(y_test, y_pred,zero_division=0),2)
    f1 = round(f1_score(y_test, y_pred,zero_division=0),2)
    kappa= round(cohen_kappa_score(y_test, y_pred),2)

    results_file.write(f"Accuracy: {accuracy:.2f}\n")
    results_file.write(f"Precision: {precision:.2f}\n")
    results_file.write(f"Recall: {recall:.2f}\n")
    results_file.write(f"F1 Score: {f1:.2f}\n")
    results_file.write(f"cohen_kappa Score: {kappa:.2f}\n")

    return accuracy,precision,recall,f1,kappa

config=conf_init()
models=config['models']
domains=config['domains']
input_path=config['CMR2_evaluated_data_dir']

results_file_name=config['CMR2_evaluated_data_dir']+"models_eval.txt"
results_file=open(results_file_name,'w')


genal_agreement=[]

agreement_on_domain0=[]
agreement_on_domain1=[]
agreement_on_domain2=[]
agreement_on_domain3=[]
agreement_on_domain4=[]
agreement_on_domain5=[]
agreement_on_domain6=[]
agreement_on_domain7=[]

agreement_on_domains=[
    agreement_on_domain0,
    agreement_on_domain1,
    agreement_on_domain2,
    agreement_on_domain3,
    agreement_on_domain4,
    agreement_on_domain5,
    agreement_on_domain6,
    agreement_on_domain7,

]


agreement_with_data_generation_model0=[]
agreement_with_data_generation_model1=[]
agreement_with_data_generation_model2=[]
agreement_with_data_generation_model3=[]

agreement_with_data_generation_model=[
    agreement_with_data_generation_model0,
    agreement_with_data_generation_model1,
    agreement_with_data_generation_model2,
    agreement_with_data_generation_model3,
]

for model in models:


    df=pd.read_csv(input_path+f"{model}_model_prediction.csv")


    y_pred=df["Predicted Interaction value"]

    y_test=df["Generated Interaction value"]

    accuracy,precision,recall,f1,kappa=get_metrics(y_pred,y_test,results_file)
    genal_agreement.append(kappa)

    for i in range (len(domains)):


        tmp=df[df['Domain']==domains[i]]


        y_pred=tmp["Predicted Interaction value"]

        y_test=tmp["Generated Interaction value"]

        accuracy,precision,recall,f1,kappa=get_metrics(y_pred,y_test,results_file)
        agreement_on_domains[i].append(kappa)

    for j in range (len(models)):
        
        
        tmp=df[df['Data Generation Model']==models[j]]


        y_pred=tmp["Predicted Interaction value"]

        y_test=tmp["Generated Interaction value"]

        accuracy,precision,recall,f1,kappa=get_metrics(y_pred,y_test,results_file)

        agreement_with_data_generation_model[j].append(kappa)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

# Plot each y list with the x list
plt.plot(models, genal_agreement, label='Agreement with generated data', marker='$G$')
plt.plot(models, agreement_with_data_generation_model[0], label=f'Agreement with generated data by {models[0]}', marker='$L$')
plt.plot(models, agreement_with_data_generation_model[1], label=f'Agreement with generated data by {models[1]}', marker='$M$')
plt.plot(models, agreement_with_data_generation_model[2], label=f'Agreement with generated data by {models[2]}', marker='$G3$')
plt.plot(models, agreement_with_data_generation_model[3], label=f'Agreement with generated data by {models[3]}', marker='$G4$')



# Add labels and title
plt.xlabel('Prediction model name')
plt.ylabel("Cohen's kappa")

plt.ylim(0, 0.9) 
# Show legend
plt.legend()
plt.savefig(input_path+"figs/CMR2 Agreement between data generate model and prediction model.png", dpi=600)
# Show the plot
# Show the plot
plt.show()


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))


plt.plot(models, agreement_on_domains[0], label=f'Agreement with generated data for {domains[0]}', marker='$U$')
plt.plot(models, agreement_on_domains[1], label=f'Agreement with generated data for {domains[1]}', marker='$P$')
plt.plot(models, agreement_on_domains[2], label=f'Agreement with generated data for {domains[2]}', marker='$H$')
plt.plot(models, agreement_on_domains[3], label=f'Agreement with generated data for {domains[3]}', marker='$SM$')
plt.plot(models, agreement_on_domains[4], label=f'Agreement with generated data for {domains[4]}', marker='$C$')
plt.plot(models, agreement_on_domains[5], label=f'Agreement with generated data for {domains[5]}', marker='$S$')
plt.plot(models, agreement_on_domains[6], label=f'Agreement with generated data for {domains[6]}', marker='$P$')
plt.plot(models, agreement_on_domains[7], label=f'Agreement with generated data for {domains[7]}', marker='$F$')




# Add labels and title
plt.xlabel('Prediction model name')
plt.ylabel("Cohen's kappa")
plt.ylim(0, 0.9) 

# Show legend
plt.legend()
plt.savefig(input_path+"figs/CMR2 Agreement based on the domain.png", dpi=600)
# Show the plot
plt.show()
