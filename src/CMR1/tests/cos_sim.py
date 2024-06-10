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
input_path=config['CMR1_evaluated_data_dir']

results_file_name=config['CMR1_evaluated_data_dir']+"cos_sim.txt"
results_file=open(results_file_name,'w')

embedding_model='text-embedding-3-large'

embedding_cos_sim_df=pd.read_csv(input_path+f"cos_sim_{embedding_model}.csv")

thresholds=[]
general_results_compared_to_generated_data=[]
general_results_compared_to_predicted_data0=[]
general_results_compared_to_predicted_data1=[]
general_results_compared_to_predicted_data2=[]
general_results_compared_to_predicted_data3=[]

general_results_compared_to_generated_data0=[]
general_results_compared_to_generated_data1=[]
general_results_compared_to_generated_data2=[]
general_results_compared_to_generated_data3=[]

for i in [2,3,3.5,4,4.5,5,6,7,8]: ## iterate over the threshold 
    
    threshold=0.1*i
    thresholds.append(threshold)
    
    embedding_cos_sim_df['threshold']=embedding_cos_sim_df.apply(lambda row: row['text1_text2_cos_sim_text-embedding-3-large']>(threshold),axis=1)
    
    y_pred=embedding_cos_sim_df['threshold']
    y_test=embedding_cos_sim_df["Same Causal Variable"]

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

    generation_models=models[0]    
    tmp=embedding_cos_sim_df[embedding_cos_sim_df['model Name']==generation_models]
        
    y_pred=tmp['threshold']
    y_test=tmp["Same Causal Variable"]


    results_file.write(f"The obtained results based on cosine similarity of threshold {threshold} over {generation_models} generating model and all domain:\n")
    accuracy,precision,recall,f1,kappa=get_metrics(y_pred,y_test,results_file)
    
    general_results_compared_to_generated_data0.append(kappa)


    generation_models=models[1]    
    tmp=embedding_cos_sim_df[embedding_cos_sim_df['model Name']==generation_models]
        
    y_pred=tmp['threshold']
    y_test=tmp["Same Causal Variable"]


    results_file.write(f"The obtained results based on cosine similarity of threshold {threshold} over {generation_models} generating model and all domain:\n")
    accuracy,precision,recall,f1,kappa=get_metrics(y_pred,y_test,results_file)
    
    general_results_compared_to_generated_data1.append(kappa)


    generation_models=models[2]    
    tmp=embedding_cos_sim_df[embedding_cos_sim_df['model Name']==generation_models]
        
    y_pred=tmp['threshold']
    y_test=tmp["Same Causal Variable"]


    results_file.write(f"The obtained results based on cosine similarity of threshold {threshold} over {generation_models} generating model and all domain:\n")
    accuracy,precision,recall,f1,kappa=get_metrics(y_pred,y_test,results_file)
    
    general_results_compared_to_generated_data2.append(kappa)


    generation_models=models[3]    
    tmp=embedding_cos_sim_df[embedding_cos_sim_df['model Name']==generation_models]
        
    y_pred=tmp['threshold']
    y_test=tmp["Same Causal Variable"]


    results_file.write(f"The obtained results based on cosine similarity of threshold {threshold} over {generation_models} generating model and all domain:\n")
    accuracy,precision,recall,f1,kappa=get_metrics(y_pred,y_test,results_file)
    
    general_results_compared_to_generated_data3.append(kappa)

    #for prediction_models in models: ## iterate over the prediction_models 
    prediction_models=models[0]
    prediction_models_df=pd.read_csv(input_path+f"{prediction_models}_model_prediction.csv")

    y_pred=embedding_cos_sim_df['threshold']
    y_test=prediction_models_df["Predicted Same Causal Variable"]


    results_file.write(f"The obtained results based on cosine similarity of threshold {threshold} over {prediction_models} prediction model and all domain:\n")
        
    accuracy,precision,recall,f1,kappa=get_metrics(y_pred,y_test,results_file)
    general_results_compared_to_predicted_data0.append(kappa)
    
    prediction_models=models[1]
    prediction_models_df=pd.read_csv(input_path+f"{prediction_models}_model_prediction.csv")

    y_pred=embedding_cos_sim_df['threshold']
    y_test=prediction_models_df["Predicted Same Causal Variable"]


    results_file.write(f"The obtained results based on cosine similarity of threshold {threshold} over {prediction_models} prediction model and all domain:\n")
        
    accuracy,precision,recall,f1,kappa=get_metrics(y_pred,y_test,results_file)
    general_results_compared_to_predicted_data1.append(kappa)
    
    prediction_models=models[2]
    prediction_models_df=pd.read_csv(input_path+f"{prediction_models}_model_prediction.csv")

    y_pred=embedding_cos_sim_df['threshold']
    y_test=prediction_models_df["Predicted Same Causal Variable"]


    results_file.write(f"The obtained results based on cosine similarity of threshold {threshold} over {prediction_models} prediction model and all domain:\n")
        
    accuracy,precision,recall,f1,kappa=get_metrics(y_pred,y_test,results_file)
    general_results_compared_to_predicted_data2.append(kappa)

    prediction_models=models[3]
    prediction_models_df=pd.read_csv(input_path+f"{prediction_models}_model_prediction.csv")

    y_pred=embedding_cos_sim_df['threshold']
    y_test=prediction_models_df["Predicted Same Causal Variable"]


    results_file.write(f"The obtained results based on cosine similarity of threshold {threshold} over {prediction_models} prediction model and all domain:\n")
        
    accuracy,precision,recall,f1,kappa=get_metrics(y_pred,y_test,results_file)
    general_results_compared_to_predicted_data3.append(kappa)
    '''
        for domain in domains: ## iterate over the domains 
        
            tmp1=embedding_cos_sim_df[embedding_cos_sim_df['domain']==domain]
            tmp2=prediction_models_df[prediction_models_df['Domain']==domain]
            
            y_pred=tmp['threshold']
            y_test=tmp["Same Causal Variable"]

            results_file.write(f"The obtained results based on cosine similarity of threshold {threshold} over {prediction_models} prediction model and {domain} domain:\n")
            accuracy,precision,recall,f1,kappa=get_metrics(y_pred,y_test,results_file)'''

print(general_results_compared_to_predicted_data0)
print(general_results_compared_to_predicted_data1)
print(general_results_compared_to_predicted_data2)
print(general_results_compared_to_predicted_data3) 
print(general_results_compared_to_generated_data)
print(thresholds)


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

# Plot each y list with the x list
plt.plot(thresholds, general_results_compared_to_generated_data, label='Agreement with generated data', marker='o')
plt.plot(thresholds, general_results_compared_to_predicted_data0, label=f'Agreement with prediction of {models[0]}', marker='$L$')
plt.plot(thresholds, general_results_compared_to_predicted_data1, label=f'Agreement with prediction of {models[1]}', marker='$M$')
plt.plot(thresholds, general_results_compared_to_predicted_data2, label=f'Agreement with prediction of {models[2]}', marker='$G3$')
plt.plot(thresholds, general_results_compared_to_predicted_data3, label=f'Agreement with prediction of {models[3]}', marker='$G4$')


# Add labels and title
plt.xlabel('Cosine similarity threshold')
plt.ylabel("Cohen's kappa")


# Show legend
plt.legend()
plt.ylim(0, 0.8) 
# Show the plot

plt.savefig(input_path+"figs/CMR1 Agreement based cosine similarity and predation models.png", dpi=600)
plt.show()





import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

# Plot each y list with the x list
plt.plot(thresholds, general_results_compared_to_generated_data, label='Agreement with generated data', marker='o')
plt.plot(thresholds, general_results_compared_to_generated_data0, label=f'Agreement with generated data by {models[0]}', marker='$L$')
plt.plot(thresholds, general_results_compared_to_generated_data1, label=f'Agreement with generated data by  {models[1]}', marker='$M$')
plt.plot(thresholds, general_results_compared_to_generated_data2, label=f'Agreement with generated data by  {models[2]}', marker='$G3$')
plt.plot(thresholds, general_results_compared_to_generated_data3, label=f'Agreement with generated data by  {models[3]}', marker='$G4$')


# Add labels and title
plt.xlabel('Cosine similarity threshold')
plt.ylabel("Cohen's kappa")

plt.ylim(0, 0.8) 
# Show legend
plt.legend()
plt.savefig(input_path+"figs/CMR1 Agreement based cosine similarity and generated data by models.png", dpi=600)
# Show the plot
plt.show()
