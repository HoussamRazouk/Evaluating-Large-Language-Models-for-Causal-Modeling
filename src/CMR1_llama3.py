# This example is the new way to use the OpenAI lib for python
import sys
sys.path.append('.')
from src.init import init_lama# just sets the API key as os variable 
client=init_lama()
First_text="Vacant Buildings"
Second_text="Unoccupied Housing (Empties)"
completion = client.chat.completions.create(
    model="llama-13b-chat",
    messages=[
        {"role": "system",
         "content":"""
        You are an expert in causality and urban studies. Your task is to help users model their domain knowledge by identifying if two texts describe the same causal variable. 
        Texts that describe different values or the same value of a causal variable should be indicated."""},
        {"role": "user", "content": f"""  
                  Your task is to assess if the following two texts belong to the same causal variable. 
                        - If the two texts belong to the same causal variable, provide the variable name.
                        - If the two texts are similar but do not belong to the same variable set the variable name to '', provide your explanation.
                    Structure your answer as a JSON object including string 'Text1', string 'Text2', boolean 'Same Causal Variable', string 'Variable Name', and string 'Explanation'.
                    
                    First text: ```{First_text}```
                    Second text: ```{Second_text}```
        """}    
    ],
    response_format={ "type": "json_object" },
    temperature=0 ## no creativity here 

)

response=completion.choices[0].message.content
print(response)