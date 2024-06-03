
import json 

def CMR1_predict_sample(row,client,model):

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
    response=json.loads(response)
    response['Prediction Model']=model
    response['Generated Same Causal Variable']=row['Same Causal Variable']
    response['Data Generation Model']=row['model Name']
    response['Generated Variable Name']=row['Variable Name']
    response['Domain']=domain
    return response

### test the function 

if False:
    from openai import OpenAI
    import sys
    sys.path.append('.')
    model="gpt-3.5-turbo"
    client = OpenAI()
    row={'Text1':'High-Crime Area ','Text2':'Disaster-Prone Region','Same Causal Variable':True,
         'Variable Name':'Public Safety','model Name':'llama3-70b','domain':'urban studies'}
    response=CMR1_predict_sample(row,client,model)
    print(response)

