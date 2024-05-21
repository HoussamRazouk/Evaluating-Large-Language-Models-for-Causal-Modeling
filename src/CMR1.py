

from openai import OpenAI
from src.init import init# just sets the API key as os variable 
from data.events import load_events
import json
import pandas as pd


cognitive_map=load_events()

init()# just sets the API key as os variable 
client = OpenAI()
result=[]
models=[ "gpt-4-turbo", "gpt-3.5-turbo"]


for model in models: ### loop over the models
  
  for First_index in range(len(cognitive_map)): ## loop over the values of the cognitive map
      for Second_index in range(First_index+1,len(cognitive_map)): ## nested loop over the values of the cognitive map
          #if First_index!=Second_index: ## don't compare the same value
            First_text=cognitive_map[First_index]
            Second_text=cognitive_map[Second_index]

            completion = client.chat.completions.create(
              model=model,
              messages=[
                {"role": "system", 
                "content":"""You are an expert in the filed of causality and urban studies  that helps users to model their domain knowledge by identify if two text belong to the same causal variable
                text that describe different values or even the same value of a causal variable should be indicated
                """},
                {"role": "user", "content": f"""  
                    
                    Your task is to assess if the following two text belong to the same causal variable 
                    If the two texts belong to the same causal variable provide the variable name.
                    If the two texts are similar however do not belong to the same variable set the variable name and provide your explanation.

                    
                    First text: ```{First_text}```
                    Second text: ```{Second_text}```
                    
                    structure your answer  only as json format including 'Text1' , 'Text2', 'Same Causal Variable', 'Variable Name' and 'Explanation'
                        
                    """}
                ],
                response_format={ "type": "json_object" }
                ,
              temperature=0 ## no creativity here 
            )
            response=completion.choices[0].message.content
            response=json.loads(response)
            response['Model']=model
            
            print(response) 
            result.append(response) # collect the results 

            pd.DataFrame(result).to_csv('results/CMR1.csv',index=False)





