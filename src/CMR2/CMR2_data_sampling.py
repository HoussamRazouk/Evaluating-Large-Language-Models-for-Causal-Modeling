import pandas as pd
import random
import sys
sys.path.append('.')
from src.CMR2.config import conf_init

'''
This code construct the positive and negative examples for CMR2 

'''
random.seed(10)


config=conf_init()

domains=config['domains']

models=config['models']



number_of_positive_samples=config['number_of_positive_samples']
number_of_negative_samples=config['number_of_negative_samples']
CMR2_generated_data_dir=config['CMR2_generated_data_dir']
CMR2_sample_data_file=config['CMR2_sample_data_file']

data_set=[]

for model in models:

    for domain in domains:
        positive_examples=[]
        negative_examples=[]
        print(model)
        print(domain)

        df=pd.read_pickle(CMR2_generated_data_dir+f'CMR2_Generated_data_{model}_{domain}.pkl')
        counter=150
        while True:
            ## select a random interaction_value
            First_interaction_value_index=random.randint(0, len(df)-1)
            First_interaction_value=df['Interaction value'][First_interaction_value_index]
            if len(positive_examples) < number_of_positive_samples:
                ## sampling a positive example
                
                Variable_values=list(pd.DataFrame(df['Variable values'][First_interaction_value_index])['Variable definition'].values)
                if ((tuple([First_interaction_value,tuple(Variable_values)]) not in positive_examples)):
                    
                    positive_examples.append(tuple([First_interaction_value,tuple(Variable_values)]))
                    

            if len(negative_examples) < number_of_negative_samples:
                
                ## sampling a negative example
                
                second_interaction_value_index=random.randint(0, len(df)-1)
                
                if second_interaction_value_index!=First_interaction_value_index:
                    
                    Variable_values=list(pd.DataFrame(df['Variable values'][second_interaction_value_index])['Variable definition'].values)
                    
                    if (tuple([First_interaction_value,tuple(Variable_values)]) not in negative_examples):
                        
                        negative_examples.append(tuple([First_interaction_value,tuple(Variable_values)]))
                        
            counter= counter-1
            if  counter==0:
                break          
            if ((not(len(negative_examples) < number_of_negative_samples))and (not(len(positive_examples) < number_of_positive_samples))):
                
                break


        for i in range(len(positive_examples)):
            data_set.append({
                'Value':positive_examples[i][0],
                'Variable definition':list(positive_examples[i][1]),
                'Interaction Value':True,
                'model Name':model,
                'domain':domain
            })

        for i in range(len(negative_examples)):
                data_set.append({#Text1,Text2,Same Causal Variable,Variable Name,Explanation,Model
                'Value':negative_examples[i][0],
                'Variable definition':list(negative_examples[i][1]),
                'Interaction Value':False,
                'model Name':model,
                'domain':domain
            })




results=pd.DataFrame(data_set)
results.to_csv(CMR2_sample_data_file,index=False)


## for testing 
if False:
    import pandas as pd
    import sys
    sys.path.append('.')
    model="llama3-70b"
    CMR2_generated_data_dir="results/CMR2/CMR2_generated_data/"
    domain="urban studies"
    df=pd.read_pickle(CMR2_generated_data_dir+f'CMR2_Generated_data_{model}_{domain}.pkl')



    list(pd.DataFrame(df['Variable values'][0])['Variable definition'].values)
