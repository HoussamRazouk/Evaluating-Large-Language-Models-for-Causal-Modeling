
import json 

def CMR1_predict_sample(row,client,model):
    """
    The function provided is designed to assess whether two given texts belong to the same causal variable. 
    Below is a summary of the function's key actions and inputs:
      Generating Completion: It uses the LLM model through the client to generate a completion for the given task.
      Extracting Response: The function extracts the response from the generated completion and converts it into a JSON object.
      Adding Additional Information: It adds additional information to the response, including the prediction model, whether the generated and input causal variables are the same, the data generation model, the generated variable name, and the domain.
      Returning Response: Finally, the function returns the response as a JSON object.
    """
  

    First_text=row['Text1']
    Second_text=row['Text2']
    domain=row['domain']
    completion = client.chat.completions.create(
                model=model,
                messages=[
                  {"role": "system", 
                  "content":f"""
                  You are an expert in causality and {domain}. Your task is to help users model their domain knowledge by identifying if two texts describe the same causal variable. 
                  Texts that describe different values or the same value of a causal variable should be indicated.
                  """},
                  {"role": "user", "content": f"""  
                  Your task is to assess if the following two texts belong to the same causal variable. 
                        - If the two texts belong to the same causal variable, provide the variable name.
                        - If the two texts are similar but do not belong to the same variable set the variable name to '', provide your explanation.
                    Structure your answer as a JSON object including string 'Text1', string 'Text2', boolean 'Predicted Same Causal Variable', string 'Predicted Variable Name', and string 'Explanation'.
                    
                    First text: ```{First_text}```
                    Second text: ```{Second_text}```       
                    """}
                  ],
                  response_format={ "type": "json_object" }
                  ,
                temperature=0, ## no creativity here 
                max_tokens=4096
              )
    response=completion.choices[0].message.content
    try:
      response=json.loads(response)
        
    except:
       with open('to_check.txt','w') as f:
            f.write(str(response)+'\n')
            f.write('Prediction Model: '+str(model)+'\n')
            f.write('Generated Interaction value: '+str(row['Same Causal Variable'])+'\n')
            f.write('Data Generation Model: '+str(row['model Name'])+'\n')
            f.write('Domain: '+str(domain)+'\n')
       
    response['Prediction Model']=model
    response['Generated Same Causal Variable']=row['Same Causal Variable']
    response['Data Generation Model']=row['model Name']
    response['Generated Variable Name']=row['Variable Name']
    response['Domain']=domain
    return response
### test the function 

if False:
  import sys
  import pandas as pd
  sys.path.append('.')
  from src.init import init# just sets the API key as os variable 
  from src.init import init_lama
  from openai import OpenAI
  import sys
  sys.path.append('.')
  model="mixtral-8x22b-instruct"
  client = init_lama()
  #SMTP,Tree,False,False,,,gpt-3.5-turbo,llama3-70b,computer science,
  row={'Text1':'SMTP','Text2':'Tree','Same Causal Variable':False,
         'Variable Name':'','model Name':'gpt-3.5-turbo','domain':'computer science'}
  response=CMR1_predict_sample(row,client,model)
  #print(response)
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
  print(df)
