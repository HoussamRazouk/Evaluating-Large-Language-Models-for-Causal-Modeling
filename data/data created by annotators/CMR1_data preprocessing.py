import pandas as pd

import itertools

import editdistance
import random
from random import randrange
from tqdm.auto import tqdm
tqdm.pandas()

def get_positive_and_negative_samples(df_label_labels, number_of_samples=3):
    positive_samples=[]
    causal_variables=[]
    causal_variable_index=0
    for item in df_label_labels['Values']:
        causal_variable=df_label_labels['Causal Variable'][causal_variable_index]
        causal_variable_index+=1
        if len(item)>1:
            combinations=list(itertools.combinations(item,2))
            print(combinations)
            random.shuffle(combinations)
            print(combinations)
            number_of_appended_samples=0
            for combination in combinations:
                if editdistance.eval(combination[0],combination[1])>1:
                    positive_samples.append(combination)
                    causal_variables.append(causal_variable)
                    number_of_appended_samples+=1
                    if number_of_appended_samples==number_of_samples:
                        break

    negative_samples=[]
    number_of_appended_negative_samples=0

    while True:
        index1=randrange(len(df_label_labels['Values']))
        index2=randrange(len(df_label_labels['Values']))
        
        if index1!=index2:
            labels1=df_label_labels['Values'][index1]
            labels2=df_label_labels['Values'][index2]
            
            sample=(labels1[randrange(len(labels1))],labels2[randrange(len(labels2))])
            sample_rev=(sample[1],sample[0])
            
            if sample in negative_samples or sample_rev in negative_samples:
                continue
            else:
                negative_samples.append(sample)
                number_of_appended_negative_samples+=1
                if number_of_appended_negative_samples==len(positive_samples):
                    break

    return positive_samples, causal_variables,negative_samples

def process_samples_to_df(positive_samples, causal_variables,negative_samples):
    sampled_data=[]
    for i in range(len(negative_samples)):
        #Text1,Text2,Same Causal Variable,Variable Name,model Name,domain
        Text1=negative_samples[i][0]
        Text2=negative_samples[i][1]
        Same_Causal_Variable=False
        domain='annotators'
        sampled_data.append({
           'Text1':Text1,
           'Text2':Text2, 
           'Same Causal Variable':Same_Causal_Variable, 
           'domain':domain,
           'Variable Name': ''
        })
        
        Text1=positive_samples[i][0]
        Text2=positive_samples[i][1]
        Same_Causal_Variable=True
        domain='annotators'
        sampled_data.append({
           'Text1':Text1,
           'Text2':Text2, 
           'Same Causal Variable':Same_Causal_Variable, 
           'domain':domain,
           'Variable Name': causal_variables[i]       
        })
        
    sampled_data_df=pd.DataFrame(sampled_data)
    return sampled_data_df
    
            
            
def main():
    file_path = "data/data created by annotators/combined.csv"
    df_variables_values =  pd.read_csv(file_path,)
    
    df_variables_values['Values'] = df_variables_values['Values'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    
    Selected_Causal_Variables = ['Coach', 'Team Motivation', 'Fitness Level', 'Warm Up Exercises', 'Genetics', 'Connective Tissue Disorder', 'Contact sport', 'Tissue Weakness', 'Neuromuscular Fatigue', 'Injury']
    
    for index, row in df_variables_values.iterrows():
        if row['Causal Variable'] not in Selected_Causal_Variables:
            df_variables_values.drop(index, inplace=True)
    df_variables_values= df_variables_values.reset_index(drop=True)
    
    positive_samples, causal_variables,negative_samples = get_positive_and_negative_samples(df_variables_values, number_of_samples=3)
    
    sampled_data_df = process_samples_to_df(positive_samples, causal_variables,negative_samples)
    
    sampled_data_df.to_csv('data/data created by annotators/CMR1_positive_negative_examples_combined.csv', index=False)
    