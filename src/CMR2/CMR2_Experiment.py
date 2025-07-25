from openai import OpenAI
import sys
sys.path.append('.')
from src.init import init# just sets the API key as os variable 
from src.init import init_lama# just sets the API key as os variable
from src.CMR2.CMR2_predict_sample import CMR2_predict_sample
from src.CMR2.config import conf_init


import pandas as pd
import os


import threading


def model_thread(test_data,model,client,output_path):

    print(f"running {model}")
    
    results=[]
    for index, row in test_data.iterrows():
        print(index)
        try:
        #if True:
            response=CMR2_predict_sample(row,client,model)
            results.append(response)
            print(f"Response for {index} is {response}")
            df=pd.DataFrame(results)
            df=df[[
                'Text',
                'Variables',
                'Generated Interaction value',
                'Predicted Interaction value',
                'Data Generation Model',
                'Prediction Model',
                'Domain',
                'Explanation', 
                ]]
            df.to_csv(output_path+f"{model}_model_prediction_differences.csv",index=False)
        except:
            print("Failed")
            results=results[:-1]  # remove the last response
            print(response)
            with open(output_path+f'to_check/{model}/'+f"{model}_{index}_fail.txt",'w') as f:
                 f.write(str(response))
        
    

## read the data

def main():
    config=conf_init()
    models=config['models']
    models=["llama3-70b",
                "mixtral-8x22b-instruct",
                "gpt-3.5-turbo",
                "gpt-4-turbo"]

    input_file=config['CMR2_sample_data_file']

    input_file="results/CMR2/sampled_data/difference.csv"

    output_path=config['CMR2_evaluated_data_dir']

    test_data=pd.read_csv(input_file)
    threads = []

    for model in models:
        
        try:
            os.makedirs(output_path+f'to_check/{model}')
        except:
            print(output_path+f'to_check/{model} already exists')
        if os.path.isfile(output_path+f"{model}_model_prediction_differences.csv"):
                print(f"{model} already tested")
                continue
        if model in ["gpt-3.5-turbo","gpt-4-turbo"]: # deferent API key 
            #different API Are used 
            init()
            client = OpenAI()
        
        else :
            #different API Are used 
            client=init_lama()
        thread = threading.Thread(target=model_thread, args=(test_data,model,client,output_path))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print("All threads have finished execution.")       

