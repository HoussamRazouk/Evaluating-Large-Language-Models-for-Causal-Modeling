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

def get_metrics(y_pred,y_test,results_file):
    
    precision = round(precision_score(y_test, y_pred,zero_division=0),2)
    recall = round(recall_score(y_test, y_pred,zero_division=0),2)
    f1 = round(f1_score(y_test, y_pred,zero_division=0),2)
    kappa= round(cohen_kappa_score(y_test, y_pred),2)

    results_file.write(f"Precision: {precision:.2f}\n")
    results_file.write(f"Recall: {recall:.2f}\n")
    results_file.write(f"F1 Score: {f1:.2f}\n")
    results_file.write(f"cohen_kappa Score: {kappa:.2f}\n")

    return precision,recall,f1,kappa

config=conf_init()
models=config['models']
domains=config['domains']
input_path=config['CMR1_evaluated_data_dir']

results_file_name=config['CMR1_evaluated_data_dir']+"models_eval.txt"
results_file=open(results_file_name,'w')


genal_agreement=[]
genal_p=[]
genal_f=[]
genal_r=[]

models=config['models']
models=[
 models[3],
 models[0],
 models[1],
 models[2],
 models[4],
 models[5],
 models[6],
]

for model in models:


            df=pd.read_csv(input_path+f"{model}_model_prediction_large.csv")


            y_pred=df["Predicted Same Causal Variable"].apply(bool)

            y_test=df["Generated Same Causal Variable"].apply(bool)
            
            precision,recall,f1,kappa=get_metrics(y_pred,y_test,results_file)
            genal_agreement.append(kappa)
            genal_p.append(precision)
            genal_f.append(f1)
            genal_r.append(recall)

pd.DataFrame(data={
        'Model Name':models,
        'F1 Score':genal_f,
        'Precision':genal_p,
        'Recall':genal_r,
        "Cohen's kappa Score":genal_agreement
        
})



agreement_on_domains=[
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],

]
tmp_models=[]
tmp_models.append(models[0])
tmp_models.append(models[1])
tmp_models.append(models[2])
tmp_models.append(models[4])

for model in tmp_models:


            df=pd.read_csv(input_path+f"{model}_model_prediction_large.csv")



            for i in range (len(domains)):


                    tmp=df[df['Domain']==domains[i]]


                    y_pred=tmp["Predicted Same Causal Variable"].apply(bool)

                    y_test=tmp["Generated Same Causal Variable"].apply(bool)

                    precision,recall,f1,kappa=get_metrics(y_pred,y_test,results_file)
                    agreement_on_domains[i].append(int(kappa*100))

pd.DataFrame(data={
        'Model Name':tmp_models,
        str(domains[0]):agreement_on_domains[0],
        str(domains[1]):agreement_on_domains[1],
        str(domains[2]):agreement_on_domains[2],
        str(domains[3]):agreement_on_domains[3],
        str(domains[4]):agreement_on_domains[4],
        str(domains[5]):agreement_on_domains[5],
        str(domains[6]):agreement_on_domains[6],
        str(domains[7]):agreement_on_domains[7]
        
}).transpose()                   


agreement_with_data_generation_model=[
    [],
    [],
    [],
    [],
    [],
    [],
    [],
]
tmp_models=[]
tmp_models.append(models[0])
tmp_models.append(models[1])
tmp_models.append(models[2])
tmp_models.append(models[4])

for model in tmp_models:

    df=pd.read_csv(input_path+f"{model}_model_prediction_large.csv")
    for j in range (len(tmp_models)):
        
        
        tmp=df[df['Data Generation Model']==tmp_models[j]]


        y_pred=tmp["Predicted Same Causal Variable"]

        y_test=tmp["Generated Same Causal Variable"]

        precision,recall,f1,kappa=get_metrics(y_pred,y_test,results_file)

        agreement_with_data_generation_model[j].append(int(kappa*100))

pd.DataFrame(data={
        'Model Name':tmp_models,
        str(tmp_models[0]):agreement_with_data_generation_model[0],
        str(tmp_models[1]):agreement_with_data_generation_model[1],
        str(tmp_models[2]):agreement_with_data_generation_model[2],
        str(tmp_models[3]):agreement_with_data_generation_model[3],
        
})




for model in models:


            df=pd.read_csv(input_path+f"{model}_model_prediction_large.csv")


            for j in range (len(models)):
                
                    tmp=df[df['Data Generation Model']==models[j]]


                    y_pred=tmp["Predicted Same Causal Variable"].apply(bool)

                    y_test=tmp["Generated Same Causal Variable"].apply(bool)

                    precision,recall,f1,kappa=get_metrics(y_pred,y_test,results_file)

                    agreement_with_data_generation_model[j].append(kappa)


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

# Plot each y list with the x list
plt.plot(models, genal_agreement, label='Agreement with generated data', marker='$G$')
plt.plot(models, agreement_with_data_generation_model[0], label=f'Agreement with generated data by {models[0]}', marker='$L$')
plt.plot(models, agreement_with_data_generation_model[1], label=f'Agreement with generated data by {models[1]}', marker='$M$')
plt.plot(models, agreement_with_data_generation_model[2], label=f'Agreement with generated data by {models[2]}', marker='$G3$')
plt.plot(models, agreement_with_data_generation_model[3], label=f'Agreement with generated data by {models[3]}', marker='$G4$')
plt.plot(models, agreement_with_data_generation_model[4], label=f'Agreement with generated data by {models[2]}', marker='$G3$')
plt.plot(models, agreement_with_data_generation_model[5], label=f'Agreement with generated data by {models[3]}', marker='$G4$')
plt.plot(models, agreement_with_data_generation_model[6], label=f'Agreement with generated data by {models[3]}', marker='$G4$')



# Add labels and title
plt.xlabel('Prediction model name')
plt.ylabel("Cohen's kappa")

plt.ylim(0, 0.7) 
# Show legend
plt.legend()
plt.savefig(input_path+"figs/CMR1 Agreement between data generate model and prediction model.png", dpi=600)
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
plt.ylim(0, 0.7) 

# Show legend
plt.legend()
plt.savefig(input_path+"figs/CMR1 Agreement based on the domain.png", dpi=600)
# Show the plot
plt.show()
