
import json 

def CMR2_predict_sample(row,client,model):
    """
    The function provided is designed to assess whether two given texts belong to the same causal variable. 
    Below is a summary of the function's key actions and inputs:
      Generating Completion: It uses the LLM model through the client to generate a completion for the given task.
      Extracting Response: The function extracts the response from the generated completion and converts it into a JSON object.
      Adding Additional Information: It adds additional information to the response, including the prediction model, whether the generated and input causal variables are the same, the data generation model, the generated variable name, and the domain.
      Returning Response: Finally, the function returns the response as a JSON object.
    """
  
    #Value,Variable definition,Interaction Value,model Name,domain  
    Value=row['Value']
    Variable_definition=row['Variable definition']
    domain=row['domain']
    completion = client.chat.completions.create(
                model=model,
                messages=[
                  {"role": "system", 
                  "content":f"""
                  You are an expert in causality and {domain}. Your task is to help users model their domain knowledge by identifying if a text describe the 
                  simultaneous occurrence values of multiple causal variables. 
                  Texts that describe different values of  different causal variable simultaneously should be indicated.
                  """},
                  {"role": "user", "content": f"""  
                  Your task is to assess if the following text belong to the provided causal variables. 
                        - If the text belong to all causal variables, the the text is and interaction value.
                        - Provide variables values.
                        - If the text does not belong to all causal variables, variables values to '',
                        - Provide your explanation.
                    Structure your answer as a JSON object including string 'Text', boolean 'Predicted Interaction value', string 'Variables', string 'Variables values', and string 'Explanation'.
                    
                    Text: ```{Value}```
                    Variables: ```{Variable_definition}```       
                    """}
                  ],
                  response_format={ "type": "json_object" }
                  ,
                temperature=0, ## no creativity here 
                max_tokens=4096
              )
    response=completion.choices[0].message.content
    try:
      json_response=json.loads(response)
      json_response['Prediction Model']=model
      json_response['Generated Interaction value']=row['Interaction Value']
      json_response['Data Generation Model']=row['model Name']
      json_response['Domain']=domain
      return json_response
    except:
      try:
        if '```' in response:
          response=response.split('```')[1]

        json_response=json.loads(response)
        json_response['Prediction Model']=model
        json_response['Generated Interaction value']=row['Interaction Value']
        json_response['Data Generation Model']=row['model Name']
        json_response['Domain']=domain
        return json_response
      except:
        with open('to_check.txt','w') as f:
          f.write(str(response)+'\n')
          f.write('Prediction Model: '+str(model)+'\n')
          f.write('Generated Interaction value: '+str(row['Interaction Value'])+'\n')
          f.write('Data Generation Model: '+str(row['model Name'])+'\n')
          f.write('Domain: '+str(domain)+'\n')

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

  row={
    'Value':'Public Art Installations',
    'Variable definition':"['Urban Planning', 'Economic Activity']",
    'Interaction Value':False,
    'model Name':'llama3-70b',
    'domain':'urban studies'}
  Value=row['Value']
  Variable_definition=row['Variable definition']
  domain=row['domain']
  completion = client.chat.completions.create(
                model=model,
                messages=[
                  {"role": "system", 
                  "content":f"""
                  You are an expert in causality and {domain}. Your task is to help users model their domain knowledge by identifying if a text describe the 
                  simultaneous occurrence values of multiple causal variables. 
                  Texts that describe different values of  different causal variable simultaneously should be indicated.
                  """},
                  {"role": "user", "content": f"""  
                  Your task is to assess if the following text belong to the provided causal variables. 
                        - If the text belong to all causal variables, the the text is and interaction value.
                        - Provide variables values.
                        - If the text does not belong to all causal variables, variables values to '',
                        - Provide your explanation.
                    Structure your answer as a JSON object including string 'Text', boolean 'Predicted Interaction value', string 'Variables', string 'Variables values', and string 'Explanation'.
                    
                    Text: ```{Value}```
                    Variables: ```{Variable_definition}```       
                    """}
                  ],
                  response_format={ "type": "json_object" }
                  ,
                temperature=0, ## no creativity here 
                max_tokens=4096
              )
  response=completion.choices[0].message.content

  response=json.loads(response)
  response['Prediction Model']=model
  response['Generated Interaction value']=row['Interaction Value']
  response['Data Generation Model']=row['model Name']
  response['Domain']=domain
  #print(response)
  df=pd.DataFrame([response])
  df=df[[
                'Text',
                'Variables',
                'Generated Interaction value',
                'Predicted Interaction value',
                'Data Generation Model',
                'Prediction Model',
                'Domain',
                'Explanation', 
                ]]
  print(df)
