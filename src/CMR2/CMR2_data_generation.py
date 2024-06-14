
from openai import OpenAI
import sys
sys.path.append('.')
from src.init import init# just sets the API key as os variable 
from src.init import init_lama# just sets the API key as os variable 
from src.CMR2.config import conf_init
import json
import pandas as pd
import os

"""
This code generates a test dataset to evaluate the effectiveness of LLM (Language Model) in identifying interaction values from causal variables. 
Each LLM is instructed to provide 10 variables from a specific domain along with examples of their values. 
The values from the same causal variable are considered as positive examples for the LLM when prompted.
Conversely, values from different variables are expected to be negative examples.
"""

config=conf_init()

number_example=config['number_example']

domains=config['domains']

models=config['models']
CMR2_generated_data_dir=config['CMR2_generated_data_dir']

for model in models:
    if model in ["gpt-3.5-turbo","gpt-4-turbo"]: # deferent API key 
        
        client =init()
    else :
        client=init_lama()
    for domain in domains :
        if os.path.isfile(CMR2_generated_data_dir+f'CMR2_Generated_data_{model}_{domain}.pkl'):
            continue
        else:
            
            system_prompt = f"As an expert in the {domain} domain, your task is to define causal variables and their corresponding values."
            user_prompt = f"""Imagine you are working with high number of causal variables within the {domain} domain.
                            Each of these variables can have multiple values described in text, providing clear indicators of the variable it represents.
                            Some values, which we'll refer to as 'Interaction values', give information about more than one variable.
                            Hence, they should be included in all the variables to which they belong. 
                            Express the value as a noun phrase that reflects the actual variable. 
                            
                            Provide '{number_example}' examples of these 'Interaction values' within the {domain} and include them in their corresponding variable and the values it represent.
                            Explain how the Interaction values represent the different values of the different variables simultaneously
                            
                            Avoid using vague values such as 'low', 'moderate', 'high', or numerical values with units.
                            Avoid using the same Interaction values as the value of it corresponding to.
                            Instead, express the value as a noun phrase that reflects the actual variable.
                            
                            Structure your response as a JSON object without additional comments. The JSON should be formatted as follows:"""+"""
                            { 'Interaction Events': 
                                [ 
                                    { 'Interaction value': '',
                                        'Variable values': 
                                        [ 
                                            { 
                                                Variable definition:'',
                                                Variable value:'', 
                                            } 
                                        ],
                                        'Explanation':'' 
                                    } 
                                ] 
                            }"""## still can be improved 
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
            #if True:
                
                jason_response=json.loads(response)
                df=pd.DataFrame(jason_response["Interaction Events"])
                df.to_csv(CMR2_generated_data_dir+f'CMR2_Generated_data_{model}_{domain}.csv',index=False) 
                df.to_pickle(CMR2_generated_data_dir+f'CMR2_Generated_data_{model}_{domain}.pkl') 
                print ('succeed: ',model)


            except:
                try:## if the model add a sentence before the jason 
                    jason_response=json.loads(response[response.index('\n'):])
                    df=pd.DataFrame(jason_response["Interaction Events"])
                    df.to_csv(CMR2_generated_data_dir+f'CMR2_Generated_data_{model}_{domain}.csv',index=False) 
                    df.to_pickle(CMR2_generated_data_dir+f'CMR2_Generated_data_{model}_{domain}.pkl') 
                    print ('succeed: ',model)

                except:

                    try:## if the model missed the last bract }
                        jason_response=json.loads(response+'}')
                        df=pd.DataFrame(jason_response["Interaction Events"])
                        df.to_csv(CMR2_generated_data_dir+f'CMR2_Generated_data_{model}_{domain}.csv',index=False) 
                        df.to_pickle(CMR2_generated_data_dir+f'CMR2_Generated_data_{model}_{domain}.pkl') 
                        print ('succeed: ',model)

                    except:
                        print (f'failed: {model} {domain}')
                        with open(CMR2_generated_data_dir+f'CMR2_Generated_data_{model}_{domain}.txt', "w") as f:
                            f.write(str(response))
