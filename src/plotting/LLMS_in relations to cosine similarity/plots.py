import seaborn as sns
import pandas as pd 
import matplotlib 
from matplotlib import pyplot as plt
from sklearn.metrics import cohen_kappa_score
import numpy as np
# Create separate heat maps for each metric
metrics = ["Kappa"]

LLMs=[  
        "Generated_data",
        '',
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "llama3-70b",
        "llama3-8b",
        "mixtral-8x22b-instruct",
        "mixtral-8x7b-instruct",
        "mistral-7b-instruct",
        
    ]

Tasks=["Task 1","Task 2"]
Tasks_folder_dict={"Task 1":"results/CMR1/evaluated_data/",
       "Task 2":"results/CMR2/evaluated_data/"}

Tasks_data_column_dict={"Task 1":"Same Causal Variable",
       "Task 2":"Interaction Value"}

fig, axs = plt.subplots(1, 2, figsize=(12, 8))

#for j, metric in enumerate(metrics):


for i, Task in enumerate(Tasks):
    
    ax = axs[i]
    embedding_cos_sim_df=pd.read_csv(Tasks_folder_dict[Task]+"cos_sim_text-embedding-3-large_large.csv")
    thresholds=np.arange(start=0, stop=1, step=0.1) 
    print(thresholds)
    Heat_map_data=pd.DataFrame(data={'Model':LLMs})
    values=[]
    for threshold in thresholds:
        
        embedding_cos_sim_df['threshold']=embedding_cos_sim_df['text1_text2_cos_sim_text-embedding-3-large']>threshold
        y_pred=embedding_cos_sim_df['threshold']
        
        for model in LLMs:
            if model=="":
                values.append(np.nan)
            else:
                if model=="Generated_data":
                    
                    y_test=embedding_cos_sim_df[Tasks_data_column_dict[Task]]
                    
                else:
                    if Task==Tasks[0]:
                        data=pd.read_csv(Tasks_folder_dict[Task]+f"{model}_model_prediction_large.csv")
                        y_test=data['Predicted Same Causal Variable'].apply(bool)
                    elif Task==Tasks[1]:
                        data=pd.read_csv(Tasks_folder_dict[Task]+f"{model}_model_prediction.csv")
                        y_test=data['Predicted Interaction value'].apply(bool)
                    
                    
                kappa= round(cohen_kappa_score(y_test, y_pred),2)*100
                values.append(kappa)
                
                #print(f"Model: {model}, Threshold: {threshold}, Kappa: {kappa}")
        
        Heat_map_data[str(int(round(threshold,2)*100))+'%']=values
        values=[]  # Reset values for the next threshold
    
    Heat_map_data=Heat_map_data.set_index(Heat_map_data["Model"])
    Heat_map_data=Heat_map_data.drop(columns=["Model"]) 
    sns.heatmap(Heat_map_data, annot=False,cbar=True,yticklabels=True, cmap="coolwarm", square=False,linewidths=0.1, linecolor="white",vmax=100,vmin=0,  ax=ax)
    
    ax.set_title(f"Models Performance on {Task} - Cohen's Kappa % vs Cosine Similarity Threshold")               
    ax.set_xlabel("Cosine Similarity Threshold")        
fig.subplots_adjust(hspace=0)
plt.tight_layout()
plt.show()    


"""      
threshold=0.5
Task="Task 1"
embedding_cos_sim_df=pd.read_csv(Tasks_folder_dict[Task]+"cos_sim_text-embedding-3-large_large.csv")
embedding_cos_sim_df['threshold']=embedding_cos_sim_df['text1_text2_cos_sim_text-embedding-3-large']>threshold
y_pred=embedding_cos_sim_df['threshold']

model="mixtral-8x7b-instruct"
data=pd.read_csv(Tasks_folder_dict[Task]+f"{model}_model_prediction_large.csv")
y_test=data['Predicted Same Causal Variable'].apply(bool)
kappa= round(cohen_kappa_score(y_test, y_pred),3)
print(f"Model: {model}, Threshold: {threshold}, Kappa: {kappa}")

model="llama3-70b"
data=pd.read_csv(Tasks_folder_dict[Task]+f"{model}_model_prediction_large.csv")
y_test=data['Predicted Same Causal Variable'].apply(bool)
kappa= round(cohen_kappa_score(y_test, y_pred),3)
print(f"Model: {model}, Threshold: {threshold}, Kappa: {kappa}")
    for j, metric in enumerate(metrics):
        
        results/CMR1/evaluated_data/cos_sim_text-embedding-3-large_large.csv
    
    for j, llm in enumerate(LLMs):
        #try:
            
            
            
            for data_set in Data_sets:
                data= pd.read_csv(f"src/plotting/LLMs on different data sets/results_{Task}_{data_set}.csv")
                HM["Model"]=data["Model"]
                HM[f"{data_set}"]=data[metric]
            HM=HM.set_index(HM["Model"])
            HM=HM[["WikiCausal","Annotators","LLMs"]]       
            #matrix = data.pivot_table(index="Model", values=metric, aggfunc="mean")
            if j==(len(metrics)-1):
                sns.heatmap(HM, annot=True,yticklabels=False, cmap="coolwarm", square=True, fmt="d",linewidths=0.6, linecolor="white",vmax=100,vmin=0,  ax=ax)
                ax.set( ylabel="")
            elif j==0:
                sns.heatmap(HM, annot=True,cbar=False,yticklabels=True, cmap="coolwarm", square=True, fmt="d",linewidths=0.6, linecolor="white",vmax=100,vmin=0,  ax=ax)
                if i==0:
                    ax.set(ylabel="Task 1")
                elif i==1:
                    ax.set(ylabel="Task 2")
                    
            else:
                sns.heatmap(HM, annot=True,cbar=False, yticklabels=False,cmap="coolwarm", square=True,fmt="d", linewidths=0.6, linecolor="white",vmax=100,vmin=0,  ax=ax)
                ax.set( ylabel="")
            titles={
                "Kappa":"Cohen's Kappa",
                "F1":"F1 Scour",
                "P":"Precision",
                "R":"Recall"
            }
            ax.set_title(f"{titles[metric]} %")
            
                
            
        #except: 
        #    print(f"results_{Task}_{data_set}")
fig.subplots_adjust(hspace=0)
plt.tight_layout()
plt.show()
"""  


