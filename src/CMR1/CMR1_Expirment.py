from openai import OpenAI
import sys
sys.path.append('.')
from src.init import init# just sets the API key as os variable 
from src.init import init_lama# just sets the API key as os variable
from src.CMR1.CMR1_predict_sample import CMR1_predict_sample
from src.CMR1.config import conf_init

import pandas as pd
import os
from tqdm.auto import tqdm
tqdm.pandas()

config=conf_init()
models=config['models']
models=["mixtral-8x22b-instruct"]

## read the data

input_file=config['CMR1_sample_data_file']

output_path=config['CMR1_evaluated_data_dir']

test_data=pd.read_csv(input_file)


for model in models:
    
    if os.path.isfile(output_path+f"{model}_model_prediction.csv"):
            print(f"{model} already tested")
            continue
    if model in ["gpt-3.5-turbo","gpt-4-turbo"]: # deferent API key 
        #different API Are used 
        init()
        client = OpenAI()
    
    else :
        #different API Are used 
        client=init_lama()
        
        
    print(f"running {model}")
    
    results=[]
    for index, row in test_data.iterrows():
        print(index)
        try:
            response=CMR1_predict_sample(row,client,model)
            results.append(response)
            df=pd.DataFrame(results)
            df=df[[
                'Text1',
                'Text2',
                'Generated Same Causal Variable',
                'Predicted Same Causal Variable',
                'Generated Variable Name',
                'Predicted Variable Name',
                'Data Generation Model',
                'Prediction Model',
                'Domain',
                'Explanation', 
                ]]
            df.to_csv(output_path+f"{model}_model_prediction.csv",index=False)
        except:
            print("Failed")
            print(row)
        
    
