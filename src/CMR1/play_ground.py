from openai import OpenAI
import sys
sys.path.append('.')
from src.init import init# just sets the API key as os variable 
from src.init import init_lama# just sets the API key as os variable
from src.CMR1.CMR1_predict_sample import CMR1_predict_sample
from src.CMR1.config import conf_init
import pandas as pd

config=conf_init()
models=config['models']
model="mistral-7b-instruct"
input_file=config['CMR1_sample_data_file']

client=init_lama()
test_data=pd.read_csv(input_file)
#1040
#626
#442
numb=1040
response=CMR1_predict_sample(test_data.iloc[numb],client,model)
df=pd.DataFrame([response])
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

df.to_csv('to_check.csv',index=False)


response={
  "Text1": "Central Processing Unit (CPU): The brain of the computer",
  "Text2": "Random Access Memory (RAM): Temporary data storage",
  "Predicted Same Causal Variable": False,
  "Predicted Variable Name": "",
  "Explanation": "The first text refers to the Central Processing Unit (CPU) as the 'brain of the computer', implying a role in processing information. The second text, on the other hand, describes Random Access Memory (RAM) as 'temporary data storage'. These texts do not refer to the same causal variable as they describe different aspects of a computer system."
}

response['Prediction Model']=model
response['Generated Same Causal Variable']=test_data.iloc[numb]['Same Causal Variable']
response['Data Generation Model']=test_data.iloc[numb]['model Name']
response['Generated Variable Name']=test_data.iloc[numb]['Variable Name']
response['Domain']=test_data.iloc[numb]['domain']

df=pd.DataFrame([response])
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

df.to_csv('to_check.csv',index=False)