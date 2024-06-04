
from openai import OpenAI
import sys
sys.path.append('.')
from src.init import init# just sets the API key as os variable 
from src.init import init_lama# just sets the API key as os variable 
from src.CMR1.config import conf_init
import json
import pandas as pd
import os

"""
This code generates a test dataset to evaluate the effectiveness of LLM (Language Model) in identifying causal variables from their values. 
Each LLM is instructed to provide 10 variables from a specific domain along with examples of their values. 
The values from the same causal variable are considered as positive examples for the LLM when prompted.
Conversely, values from different variables are expected to be negative examples.
"""

config=conf_init()

number_variables=config['number_variables']

domains=config['domains']

models=config['models']
CMR1_generated_data_dir=config['CMR1_generated_data_dir']

for model in models:
    if model in ["gpt-3.5-turbo","gpt-4-turbo"]: # deferent API key 
        
        client =init()
    else :
        client=init_lama()
    for domain in domains :
        if os.path.isfile(CMR1_generated_data_dir+f'CMR1_Generated_data_{model}_{domain}.csv'):
            continue
        else:
            
            system_prompt = f"As an expert in the {domain} domain, your task is to define causal variables and their corresponding values."
            user_prompt = f"""Imagine you have {number_variables} causal variables within the {domain} domain.
                Each of these variables can have multiple values described in text, providing clear indicators of the variable it represents. 
                Avoid using vague values such as 'low', 'moderate', 'high', or numerical values with units.
                Instead, express the value as a noun phrase that reflects the actual variable. 
                
                Please provide an example of these variables and their corresponding values, ensuring that the variables are independent from each other.
                
                Structure your response as a JSON object without additional comments.
                The JSON should include a 'variable definition' and a list of 'values' for each variable, formatted as follows:
                """+"""{
                'Variables': [
                    {
                    'variable definition': '',
                    'values': []
                    }
                ]
                }
                """ ,## still can be improved 
            completion = client.chat.completions.create(
            model=model,
            messages=[
                    {"role": "system",
                    "content":system_prompt},
                    {"role": "user", 
                    "content": user_prompt}],
                response_format={ "type": "json_object" },
                temperature=0, ## consistent answers,
                max_tokens=4096
            )

            response=completion.choices[0].message.content

            try:
                
                response=json.loads(response)
                df=pd.DataFrame(response["Variables"])
                df.to_csv(CMR1_generated_data_dir+f'CMR1_Generated_data_{model}_{domain}.csv',index=False) 
                print ('succeed: ',model)


            except:

                print(response)
                print (model)
                print ('failed: ',model)
                with open(CMR1_generated_data_dir+f'CMR1_Generated_data_{model}_{domain}.txt', "w") as f:
                    f.write(str(response))
