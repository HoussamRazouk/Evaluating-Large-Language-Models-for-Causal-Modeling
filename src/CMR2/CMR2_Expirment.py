from openai import OpenAI
import sys
sys.path.append('.')
from src.init import init# just sets the API key as os variable 
from src.init import init_lama# just sets the API key as os variable
from src.CMR2.CMR2_predict_sample import CMR2_predict_sample
from src.CMR2.config import conf_init

import logging
logging.basicConfig(filename='CMR2.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logging.info("Running Urban Planning")

import pandas as pd
import os
from tqdm.auto import tqdm
tqdm.pandas()

config=conf_init()
models=config['models']


## read the data

input_file=config['CMR2_sample_data_file']

output_path=config['CMR2_evaluated_data_dir']

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
            response=CMR2_predict_sample(row,client,model)
            results.append(response)
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
            df.to_csv(output_path+f"{model}_model_prediction.csv",index=False)
        except:
            print("Failed")
            print(row)
            logging.info(model)
            logging.info(str(row))
        
    
