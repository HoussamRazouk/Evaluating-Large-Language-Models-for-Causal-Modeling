
from openai import OpenAI
import sys
sys.path.append('.')
from src.init import init# just sets the API key as os variable 
from src.init import init_lama# just sets the API key as os variable 
import json
import pandas as pd
import os

"""
this code generate a data test to LLM effectivity in identifying causal variables from their values.
each LLM is asked to provide 10 variables from a specific domain  and example from their values.
Values from the same causal variable are considered to generate positive examples for the llm when it is prompt.
Values from different variables are expected to be negative examples.

"""


number_variables=10

domains=["urban studies","physics","health","semiconductor manufacturing","computer science","sociology","psychology"]

models=["llama3-70b","mixtral-8x22b-instruct","gpt-3.5-turbo","gpt-4-turbo"]


for model in models:
    if model in ["gpt-3.5-turbo","gpt-4-turbo"]: # deferent API key 

        client = OpenAI()
    else :
        client=init_lama()
    for i in range(len(domains)) :
        if os.path.isfile(f'results/CMR1_Generated_data_{model}_{domains[i]}.csv'):
            continue
        else:
            df=pd.DataFrame()
    #For example '''{example_variable[0]}'''' can be indicated by high number of '''{example_value[0]}'''. 
            system_prompt=f"You are an expert in  {domains[i]}. "

            user_prompt=f"""Consider you have {number_variables} variables from the {domains[i]} domain. 
            Each of these variables can be instantiated using take large number of values described in text.
            These values should give a clear indicators about the variable it describes.
            Avoid values such as '''low''' or '''moderate''' or '''high''' or '''numerical values and unites'''.
            Instead, writ the value as nouns phrase which you could understand infer the actual variable from.
            Provide an example of these variables  and their corresponding values.
            Make sure that these variables are independent from each other.
            Structure your answer only as a JSON object with no additional comment.
            The JSON response should including string 'variable definition', list 'values'.
            The answer JSON should be structured as follows
            '''
            'Variables':['variable definition':'','values':[]]
            '''            
            """## still can be improved 
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
                df=df.append(response["Variables"], ignore_index=True)
                df.to_csv(f'results/CMR1_Generated_data_{model}_{domains[i]}.csv',index=False) 
                print ('succeed: ',model)


            except:

                print(response)
                print (model)
                print ('failed: ',model)
                with open(f'results/CMR1_Generated_data_{model}_{domains[i]}.txt', "w") as f:
                    f.write(str(response))
