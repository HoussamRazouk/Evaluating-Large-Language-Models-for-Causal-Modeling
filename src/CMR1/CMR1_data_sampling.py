import pandas as pd
import random
import sys
sys.path.append('.')
from src.CMR1.config import conf_init

'''
This code construct the positive and negative examples for CMR1 

'''
random.seed(10)
config=conf_init()

domains=config['domains']

models=config['models']


number_of_positive_samples=config['number_of_positive_samples']
number_of_negative_samples=config['number_of_negative_samples']
CMR1_generated_data_dir=config['CMR1_generated_data_dir']
CMR1_sample_data_file=config['CMR1_sample_data_file']
data_set=[]

for model in models:

    for domain in domains:
        positive_examples=[]
        negative_examples=[]

        df=pd.read_csv(CMR1_generated_data_dir+f'CMR1_Generated_data_{model}_{domain}.csv',converters={'values': pd.eval})
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
                second_variable_index=random.randint(0, len(df)-1)
                if second_variable_index!=first_variable_index:
                    second_value_index=random.randint(0, len(df['values'][second_variable_index])-1)
                    if tuple([second_variable_index,
                                second_value_index,
                                first_variable_index,
                                first_value_index]) not in negative_examples:
                        negative_examples.append(tuple([second_variable_index,
                                                       second_value_index,
                                                       first_variable_index,
                                                       first_value_index]))
                        negative_examples.append(tuple([first_variable_index,
                                                       first_value_index,
                                                       second_variable_index,
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
results.to_csv(CMR1_sample_data_file,index=False)


