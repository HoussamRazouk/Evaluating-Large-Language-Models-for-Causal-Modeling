

from openai import OpenAI
import sys
sys.path.append('.')
from src.init import init# just sets the API key as os variable 
from data.events import load_events
import json
import pandas as pd


cognitive_map=load_events()

init()# just sets the API key as os variable 
client = OpenAI()

models=[ "gpt-3.5-turbo","gpt-4-turbo"]


for model in models: ### loop over the models

  df=pd.read_csv(f'results/CMR1_{model}.csv')
  for First_index in range(len(cognitive_map)): ## loop over the values of the cognitive map
      for Second_index in range(First_index+1,len(cognitive_map)): ## nested loop over the values of the cognitive map
          #if First_index!=Second_index: ## don't compare the same value
            First_text=cognitive_map[First_index]
            Second_text=cognitive_map[Second_index]
            con1=df['Text1']==First_text
            con2=df['Text2']==Second_text
            con3=df['Model']==model

            if df[(con1&con2&con3)].empty: ## to avoid doing the same query

              completion = client.chat.completions.create(
                model=model,
                messages=[
                  {"role": "system", 
                  "content":"""
                  You are an expert in causality and urban studies. Your task is to help users model their domain knowledge by identifying if two texts describe the same causal variable. 
                  Texts that describe different values or the same value of a causal variable should be indicated.
                  """},
                  {"role": "user", "content": f"""  
                  Your task is to assess if the following two texts belong to the same causal variable. 
                        - If the two texts belong to the same causal variable, provide the variable name.
                        - If the two texts are similar but do not belong to the same variable set the variable name to '', provide your explanation.
                    Structure your answer as a JSON object including string 'Text1', string 'Text2', boolean 'Same Causal Variable', string 'Variable Name', and string 'Explanation'.
                    
                    First text: ```{First_text}```
                    Second text: ```{Second_text}```       
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
              df=df.append(response, ignore_index=True)
              df.to_csv(f'results/CMR1_{model}.csv',index=False) 





