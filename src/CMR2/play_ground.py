from openai import OpenAI
import sys
sys.path.append('.')
from src.init import init# just sets the API key as os variable 
from src.init import init_lama# just sets the API key as os variable
from src.CMR2.CMR2_predict_sample import CMR2_predict_sample
from src.CMR2.config import conf_init
import pandas as pd

config=conf_init()
models=config['models']
model="mixtral-8x22b-instruct"
input_file=config['CMR2_sample_data_file']
input_file="results/CMR2/sampled_data/difference.csv"
#output_path=config['CMR2_evaluated_data_dir']
client=init_lama()
test_data=pd.read_csv(input_file)


response=CMR2_predict_sample(test_data.iloc[415],client,model)

response

df=pd.DataFrame([response])
df=df[[         'Text',
                'Variables',
                'Generated Interaction value',
                'Predicted Interaction value',
                'Data Generation Model',
                'Prediction Model',
                'Domain',
                'Explanation', 
                ]]

df.to_csv('to_check.csv',index=False)
if True:
    response={
    "Text": "Time of Check to Time of Use vulnerability in operating system",
    "Predicted Interaction value": True,
    "Variables": ["Real-time Systems", "Concurrency"],
    "Variables values": '["Real-time Systems": "Present", "Concurrency": "Present"]',
    "Explanation": "The text describes a vulnerability in operating systems that is related to both real-time systems and concurrency. Real-time systems are present because the Time of Check to Time of Use (TOCTTOU) vulnerability is a concern in systems that require real-time responses. Concurrency is also present because the TOCTTOU vulnerability arises from the race condition between the time of check and the time of use, which is a concurrency issue."
    }


    


   
    response["Prediction Model"]= "mixtral-8x22b-instruct"
    response["Generated Interaction value"]=  False
    response["Data Generation Model"]= "mistral-7b-instruct"
    response["Domain"]= "computer science"
    df=pd.DataFrame([response])
    df=df[[         'Text',
                    'Variables',
                    'Generated Interaction value',
                    'Predicted Interaction value',
                    'Data Generation Model',
                    'Prediction Model',
                    'Domain',
                    'Explanation', 
                    ]]

    df.to_csv('to_check.csv',index=False)


