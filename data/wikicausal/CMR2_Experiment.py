
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()
import sys
sys.path.append('.')
from src.CMR2.get_cos_sim import get_cos_sim
import pandas as pd
from src.CMR2.config import conf_init
from src.init import init, init_lama
from openai import OpenAI
from src.CMR2.CMR2_Experiment import model_thread
from sklearn.metrics import cohen_kappa_score, precision_score,recall_score, f1_score, accuracy_score
import os


def get_embedding_cos_sim_df(sampled_data_df, embeddings_model='text-embedding-3-large'):
    
    sampled_data_df[f'text1_text2_cos_sim_{embeddings_model}']=sampled_data_df.progress_apply(lambda row: get_cos_sim(row['Text1'],row['Text2'],embeddings_model),axis=1)
    sampled_data_df.to_csv('data/wikicausal/CMR2_positive_negative_examples_enwiki-20220901-kg_v6-corpus_v4.jsonl.csv',index=False)

    #return sampled_data_df

#emb_text2=emb_text2=openai.embeddings.create(input = '[text2]', model=embeddings_model).data[0].embedding

def run_model_on_wikicausal(model,sampled_data_df,client):
    
    output_path='data/wikicausal/test CMR2/'
    
    try:
        os.makedirs(output_path+f'to_check/{model}')
    except:
        print(output_path+f'to_check/{model} already exists')
        dir_path=output_path+f'to_check/{model}'
        for filename in os.listdir(dir_path):
            # Construct the full file path
            file_path = os.path.join(dir_path, filename)
            
            # Check if the file is a regular file (not a directory)
            if os.path.isfile(file_path):
                # Delete the file
                os.remove(file_path)
                print(f"Deleted file: {filename}")
        
    
    model_thread(sampled_data_df, model, client, output_path)
    
    
def get_metrics(results_file,model):
    test_df=results_file
    y_test= test_df["Generated Interaction value"].apply(bool)# Generated Same Causal Variable,Predicted Same Causal Variable
    y_pred= test_df["Predicted Interaction value"].apply(bool)
    kappa= round(cohen_kappa_score(y_test, y_pred),2)
    Precision= round(precision_score(y_test, y_pred),2)
    Recall= round(recall_score(y_test, y_pred),2)
    F1= round(f1_score(y_test, y_pred),2)
    print(f"F1 for {model}: {F1}")
    print(f"Precision for {model}: {Precision}")
    print(f"Recall for {model}: {Recall}")
    print(f"Kappa for {model}: {kappa}")
    

def main():
    
    config=conf_init()
    sampled_data_df=pd.read_csv('data/wikicausal/CMR2_positive_negative_examples_enwiki-20220901-kg_v6-corpus_v4.jsonl.csv')
    #sampled_data_df=get_embedding_cos_sim_df(sampled_data_df, embeddings_model='text-embedding-3-large')
    sampled_data_df['model Name']='wikicausal'
    sampled_data_df['Variable Name']=''
    sampled_data_df['domain']=''
    
    models=["llama3-70b",
            "mixtral-8x22b-instruct",
            "gpt-3.5-turbo",
            "gpt-4-turbo",
            "llama3-8b",
            "mixtral-8x7b-instruct",
            "mistral-7b-instruct"
            ]
    
    models=[
            "llama3-8b",
            ]
    
    
    
    for model in models:
        
        if model in ['gpt-3.5-turbo','gpt-4-turbo']:
            client = OpenAI()
        else:
            client=init_lama()

        run_model_on_wikicausal(model,sampled_data_df,client)
        #results_file=pd.read_csv(f'data/wikicausal/test CMR2/{model}_model_prediction_large.csv')
        #get_metrics(results_file,model)

def test():
        
    models=[
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "llama3-70b",
            "llama3-8b",
            "mixtral-8x22b-instruct",
            "mixtral-8x7b-instruct",
            "mistral-7b-instruct",
            "embedding_cos_sim"
            ]
    models=["llama3-70b",
            "llama3-8b",
            "mixtral-8x22b-instruct",
            "mixtral-8x7b-instruct",
            "mistral-7b-instruct",
            ]
    
    
    for model in models:
        if model == 'embedding_cos_sim':
            sampled_data_df=pd.read_csv('data/wikicausal/CMR2_positive_negative_examples_enwiki-20220901-kg_v6-corpus_v4.jsonl.csv')
            sampled_data_df["Generated Interaction value"]= sampled_data_df["Generated Interaction value"]
            sampled_data_df["Predicted Interaction value"]= sampled_data_df.apply(lambda row: row['text1_text2_cos_sim_text-embedding-3-large']>0.7,axis=1)
            get_metrics(sampled_data_df,model)
        else:
            results_file=pd.read_csv(f'data/wikicausal/test CMR2/{model}_model_prediction_differences.csv')
            get_metrics(results_file,model)
            
main() 

test()          
