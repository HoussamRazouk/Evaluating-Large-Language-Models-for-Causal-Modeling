# This example is the new way to use the OpenAI lib for python
import sys
sys.path.append('.')
from src.init import init,init_lama# just sets the API key as os variable 
client=init_lama()
domain="physics"
#Please provide an example of these variables and their corresponding values within the {domain}.
number_variables=10
completion = client.chat.completions.create(
    model="llama3-70b",
    messages=[
            {
        "role": "system",
        "content": f"As an expert in the {domain} domain, your task is to define causal variables and their corresponding values."
    },
    {
        "role": "user",
        "content": f"""Imagine you are working with '{number_variables}' causal variables within the {domain} domain.
        Each of these variables can have multiple values described in text, providing clear indicators of the variable it represents.
        Some values, which we'll refer to as 'Interaction values', give information about more than one variable.
        Hence, they should be included in all the variables to which they belong. Express the value as a noun phrase that reflects the actual variable. 
        Provide examples of these 'Interaction values' within the {domain} and include them in their corresponding variable values.
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
        }"""
    }    
    ],
    response_format={ "type": "json_object" },
    temperature=0, ## no creativity here 
    max_tokens=4096
)

response=completion.choices[0].message.content
print(response)