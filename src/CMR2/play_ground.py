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
model="llama3-70b"
input_file=config['CMR2_sample_data_file']

#output_path=config['CMR2_evaluated_data_dir']
client=init_lama()
test_data=pd.read_csv(input_file)
test_data
#505
#536
#623
response=CMR2_predict_sample(test_data.iloc[623],client,model)

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


