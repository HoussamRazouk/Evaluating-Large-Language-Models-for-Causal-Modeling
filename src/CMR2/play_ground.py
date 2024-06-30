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
model="mistral-7b-instruct"
input_file=config['CMR2_sample_data_file']
#input_file="results/CMR2/sampled_data/difference.csv"
#output_path=config['CMR2_evaluated_data_dir']
client=init_lama()
test_data=pd.read_csv(input_file)




#1008
#640
#229
#127
#125
#121

numb=1008
response=CMR2_predict_sample(test_data.iloc[numb],client,model)
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


response={
  "Text": "Poor sleep quality and immune system function",
  "Predicted Interaction value": False,
  "Variables": '["Quality of sleep", "Immune system function"]',
  "Variables values": '"Quality of sleep: Poor, Immune system function: """',
  "Explanation": "The text mentions two separate causal variables: 'Quality of sleep' and 'Immune system function'. It does not indicate an interaction between the two variables."
}


response["Prediction Model"]= model
response["Generated Interaction value"]= test_data.iloc[numb]["Interaction Value"]
response["Data Generation Model"]= test_data.iloc[numb]["model Name"]
response["Domain"]= test_data.iloc[numb]["domain"]
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

