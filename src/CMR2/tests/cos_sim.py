import sys
sys.path.append('.')
import pandas as pd
from src.CMR2.config import conf_init
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix
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

results_file_name=config['CMR2_evaluated_data_dir']+"cos_sim.txt"
results_file=open(results_file_name,'w')

embedding_model='text-embedding-3-large'

embedding_cos_sim_df=pd.read_csv(input_path+f"cos_sim_{embedding_model}_large.csv")

thresholds=[]
general_results_compared_to_generated_data=[]
results_compared_to_predicted_data=[]
results_compared_to_generated_data=[]

models=models=[
 models[3],
 models[0],
 models[1],
 models[2],
 models[4],
 models[5],
 models[6],
]
for model in models:
    results_compared_to_predicted_data.append([])
    results_compared_to_generated_data.append([])





for j in range(17): ## iterate over the threshold 
    
    threshold=(0.05*j)+0.1
    thresholds.append(threshold)
    
    embedding_cos_sim_df['threshold']=embedding_cos_sim_df.apply(lambda row: row['text1_text2_cos_sim_text-embedding-3-large']>(threshold),axis=1)
    
    y_pred=embedding_cos_sim_df['threshold']
    
    y_test=embedding_cos_sim_df['Interaction Value']

    results_file.write(f"The obtained results based on cosine similarity of threshold {threshold} over all the generating models and domains:\n")

    accuracy,precision,recall,f1,kappa=get_metrics(y_pred,y_test,results_file)
    general_results_compared_to_generated_data.append(kappa)
    '''
    for domain in domains: ## iterate over the domains 
        
        tmp=embedding_cos_sim_df[embedding_cos_sim_df['domain']==domain]
        
        y_pred=tmp['threshold']
        y_test=tmp["Same Causal Variable"]


        results_file.write(f"The obtained results based on cosine similarity of threshold {threshold} over all the generating models and {domain} domain:\n")

        accuracy,precision,recall,f1,kappa=get_metrics(y_pred,y_test,results_file)
    '''
    for i in range(len(models)):
        
        generation_models=models[i]    
        tmp=embedding_cos_sim_df[embedding_cos_sim_df['model Name']==generation_models]
        
        y_pred=tmp['threshold']
        y_test=tmp['Interaction Value']


        results_file.write(f"The obtained results based on cosine similarity of threshold {threshold} over {generation_models} generating model and all domain:\n")
        accuracy,precision,recall,f1,kappa=get_metrics(y_pred,y_test,results_file)
    
        results_compared_to_generated_data[i].append(kappa)


    #for prediction_models in models: ## iterate over the prediction_models
    for i in range(len(models)): 
        
        prediction_models=models[i]
        prediction_models_df=pd.read_csv(input_path+f"{prediction_models}_model_prediction.csv")

        y_pred=embedding_cos_sim_df['threshold']
        y_test=prediction_models_df["Predicted Interaction value"]


        results_file.write(f"The obtained results based on cosine similarity of threshold {threshold} over {prediction_models} prediction model and all domain:\n")
        
        accuracy,precision,recall,f1,kappa=get_metrics(y_pred,y_test,results_file)
        results_compared_to_predicted_data[i].append(kappa)
    
    '''
        for domain in domains: ## iterate over the domains 
        
            tmp1=embedding_cos_sim_df[embedding_cos_sim_df['domain']==domain]
            tmp2=prediction_models_df[prediction_models_df['Domain']==domain]
            
            y_pred=tmp['threshold']
            y_test=tmp["Same Causal Variable"]

            results_file.write(f"The obtained results based on cosine similarity of threshold {threshold} over {prediction_models} prediction model and {domain} domain:\n")
            accuracy,precision,recall,f1,kappa=get_metrics(y_pred,y_test,results_file)'''


print(general_results_compared_to_generated_data)
print(thresholds)


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

# Plot each y list with the x list
plt.plot(thresholds, general_results_compared_to_generated_data, label='Agreement with generated data', marker='o')
plt.plot(thresholds, results_compared_to_predicted_data[0], label=f'{models[0]}', marker='s')
plt.plot(thresholds, results_compared_to_predicted_data[1], label=f'{models[1]}', marker='^')
plt.plot(thresholds, results_compared_to_predicted_data[2], label=f'{models[2]}', marker='v')
plt.plot(thresholds, results_compared_to_predicted_data[3], label=f'{models[3]}', marker='d')
plt.plot(thresholds, results_compared_to_predicted_data[4], label=f'{models[4]}', marker='+')
plt.plot(thresholds, results_compared_to_predicted_data[5], label=f'{models[5]}', marker='x')
plt.plot(thresholds, results_compared_to_predicted_data[6], label=f'{models[6]}', marker='*')





# Add labels and title
plt.xlabel('Cosine similarity threshold')
plt.ylabel("Cohen's kappa")


# Show legend
plt.legend()
plt.ylim(0, 0.9) 
# Show the plot

plt.savefig(input_path+"figs/CMR2 Large Agreement based cosine similarity and predation models.png", dpi=600)
plt.show()





import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

# Plot each y list with the x list
plt.plot(thresholds, general_results_compared_to_generated_data, label='Agreement with generated data', marker='o')
plt.plot(thresholds, results_compared_to_generated_data[0], label=f'{models[0]}', marker='s')
plt.plot(thresholds, results_compared_to_generated_data[1], label=f'{models[1]}', marker='^')
plt.plot(thresholds, results_compared_to_generated_data[2], label=f'{models[2]}', marker='v')
plt.plot(thresholds, results_compared_to_generated_data[3], label=f'{models[3]}', marker='d')
plt.plot(thresholds, results_compared_to_generated_data[4], label=f'{models[4]}', marker='+')
plt.plot(thresholds, results_compared_to_generated_data[5], label=f'{models[5]}', marker='x')
plt.plot(thresholds, results_compared_to_generated_data[6], label=f'{models[6]}', marker='*')


# Add labels and title
plt.xlabel('Cosine similarity threshold')
plt.ylabel("Cohen's kappa")

plt.ylim(0, 0.9) 
# Show legend
plt.legend()
plt.savefig(input_path+"figs/CMR2 Large Agreement based cosine similarity and generated data by models.png", dpi=600)
# Show the plot
plt.show()
