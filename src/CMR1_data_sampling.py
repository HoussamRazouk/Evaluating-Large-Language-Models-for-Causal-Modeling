import pandas as pd
import random

'''
This code construct the positive and negative examples for CMR1 

'''
domains=["urban studies","physics","health","semiconductor manufacturing","computer science","sociology","psychology","finance"]
models=["llama3-70b","mixtral-8x22b-instruct","gpt-3.5-turbo","gpt-4-turbo"]

number_of_positive_samples=10
number_of_negative_samples=10
data_set=[]
for model in models:

    for domain in domains:
        positive_examples=[]
        negative_examples=[]

        df=pd.read_csv(f'results/generated_data_causal_variables_and_values/CMR1_Generated_data_{model}_{domain}.csv',converters={'values': pd.eval})
        while True:
            ## select a random variable
            first_variable_index=random.randint(0, len(df)-1)
            first_value_index=random.randint(0, len(df['values'][first_variable_index])-1)
            if len(positive_examples) < 2*number_of_positive_samples:
                ## sampling a postive example
                second_value_index=random.randint(0, len(df['values'][first_variable_index])-1)
                if ((second_value_index!=first_value_index) and(tuple([first_variable_index,second_value_index,first_variable_index,first_value_index]) not in positive_examples)):
                    positive_examples.append(tuple([first_variable_index,second_value_index,first_variable_index,first_value_index]))
                    positive_examples.append(tuple([first_variable_index,first_value_index,first_variable_index,second_value_index]))

            if len(negative_examples) < 2*number_of_negative_samples:
                ## sampling a negative example
                secund_variable_index=random.randint(0, len(df)-1)
                if secund_variable_index!=first_variable_index:
                    second_value_index=random.randint(0, len(df['values'][secund_variable_index])-1)
                    if tuple([secund_variable_index,
                                second_value_index,
                                first_variable_index,
                                first_value_index]) not in negative_examples:
                        negative_examples.append(tuple([secund_variable_index,
                                                       second_value_index,
                                                       first_variable_index,
                                                       first_value_index]))
                        negative_examples.append(tuple([first_variable_index,
                                                       first_value_index,
                                                       secund_variable_index,
                                                       second_value_index]))
            if ((not(len(negative_examples) < 2*number_of_negative_samples))and (not(len(positive_examples) < 2*number_of_positive_samples))):
                break


        for i in range(number_of_positive_samples):
            data_set.append({
                'Text1':df['values'][positive_examples[i*2][0]][positive_examples[i*2][1]],
                'Text2':df['values'][positive_examples[i*2][2]][positive_examples[i*2][3]],
                'Same Causal Variable':True,
                'Variable Name':df['variable definition'][positive_examples[i*2][0]],
                'model Name':model,
                'domain':domain
            })

        for i in range(number_of_negative_samples):
                data_set.append({#Text1,Text2,Same Causal Variable,Variable Name,Explanation,Model
                'Text1':df['values'][negative_examples[i*2][0]][negative_examples[i*2][1]],
                'Text2':df['values'][negative_examples[i*2][2]][negative_examples[i*2][3]],
                'Same Causal Variable':False,
                'Variable Name':'',
                'model Name':model,
                'domain':domain
            })




results=pd.DataFrame(data_set)
results.to_csv('results/sampled_data_set.csv',index=False)


