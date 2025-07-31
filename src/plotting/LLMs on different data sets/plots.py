import seaborn as sns
import pandas as pd 
import matplotlib 
from matplotlib import pyplot as plt

# Create separate heat maps for each metric
metrics = ["Kappa", "F1", "P", "R"]
Data_sets=["WikiCausal","Annotators","LLMs"]
Tasks=["Task 1","Task 2"]
fig, axs = plt.subplots(2, 4, figsize=(12, 8))

#for j, metric in enumerate(metrics):
    
for i, Task in enumerate(Tasks):
    
    for j, metric in enumerate(metrics):
        #try:
            ax = axs[i, j]
            HM = pd.DataFrame()
            
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



