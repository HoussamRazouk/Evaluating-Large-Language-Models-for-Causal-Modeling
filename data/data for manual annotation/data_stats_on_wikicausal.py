


import pandas as pd

import itertools

import editdistance
import random
from random import randrange
from tqdm.auto import tqdm
tqdm.pandas()

file_path="data/data for manual annotation/enwiki-20220901-kg_v6-corpus_v4.jsonl"

df = pd.read_json(file_path, orient='records', lines=True)

label=[]## we consider the event concept as the causal variable 
labels=[]## we consider the event labels as the values belonging to the same causal variable 
for item in df['event_concepts']:
    for event in item:
        label.append(event['label'])
        event['labels'].sort()
        labels.append(event['labels'])
           
df_label_labels=pd.DataFrame(data={
    'label':label,
    'labels':labels
})        
df_label_labels['labels_str']= df_label_labels.apply(lambda row: str(row['labels']),axis=1)
df_label_labels=df_label_labels.drop_duplicates(subset=['label','labels_str'],keep='last')
df_label_labels= df_label_labels.reset_index(drop=True)
df_label_labels=  df_label_labels[['label','labels']] 

### we need to remove the entries in the labels with edit distance lower than 2
for i in range(len(df_label_labels['labels'])):
    test=df_label_labels['labels'][i]
    edited=True
    while edited:
        edited=False
        combinations=list(itertools.combinations(test,2))
        for combination in combinations:
            distance=editdistance.eval(combination[0],combination[1])
            if distance<2:
                test.remove(combination[1])
                print(combination[1])
                print(combination)
                edited=True
                break
                
            
   
            


df_label_labels.to_csv('data/data for manual annotation/concepts_and_labels_enwiki-20220901-kg_v6-corpus_v4.jsonl.csv',index='False')

positive_samples=[]
number_of_samples=3

for item in df_label_labels['labels']:
    
    if len(item)>1:
        combinations=list(itertools.combinations(item,2))
        print(combinations)
        random.shuffle(combinations)
        print(combinations)
        number_of_appended_samples=0
        for combination in combinations:
            
            if editdistance.eval(combination[0],combination[1])>1:
                
                positive_samples.append(combination)
                number_of_appended_samples=number_of_appended_samples+1
                
                if number_of_appended_samples==number_of_samples:
                    break


len(positive_samples)  

negative_samples=[]
number_of_appended_negative_samples=0

while True:
    
    index1=randrange(len(df_label_labels['labels']))
    
    index2=randrange(len(df_label_labels['labels']))
    
    if index1!=index2:
        
        labels1=df_label_labels['labels'][index1]
        labels2=df_label_labels['labels'][index2]
        
        sample=(labels1[randrange(len(labels1))],labels2[randrange(len(labels2))])
        sample_rev=(sample[1],sample[0])
        
        if sample in negative_samples:
            continue
        elif sample_rev in negative_samples:
            continue
        else:
            negative_samples.append(sample)
            number_of_appended_negative_samples=number_of_appended_negative_samples+1
            if number_of_appended_negative_samples==len(positive_samples):
                break
            
            
            
len(negative_samples)
sampled_data=[]
for i in range(len(negative_samples)):
    #Text1,Text2,Same Causal Variable,Variable Name,model Name,domain
    Text1=negative_samples[i][0]
    Text2=negative_samples[i][1]
    Same_Causal_Variable=False
    domain='wiki_causal'
    sampled_data.append({
       'Text1':Text1,
       'Text2':Text2, 
       'Same Causal Variable':Same_Causal_Variable, 
       'domain':domain       
    })
    
    Text1=positive_samples[i][0]
    Text2=positive_samples[i][1]
    Same_Causal_Variable=True
    domain='wiki_causal'
    sampled_data.append({
       'Text1':Text1,
       'Text2':Text2, 
       'Same Causal Variable':Same_Causal_Variable, 
       'domain':domain       
    })
    
sampled_data_df=pd.DataFrame(sampled_data)



import sys
sys.path.append('.')
from src.CMR1.get_cos_sim import get_cos_sim
import pandas as pd
from src.CMR2.config import conf_init
from src.init import init
import openai
from openai import OpenAI
config=conf_init()



embeddings_model='text-embedding-3-large'

emb_text2=emb_text2=openai.embeddings.create(input = '[text2]', model=embeddings_model).data[0].embedding
sampled_data_df[f'text1_text2_cos_sim_{embeddings_model}']=sampled_data_df.progress_apply(lambda row: get_cos_sim(row['Text1'],row['Text2'],embeddings_model),axis=1)
sampled_data_df.to_csv('data/data for manual annotation/positive_negative_examples_enwiki-20220901-kg_v6-corpus_v4.jsonl.csv',index=False)    

 
        
        
        
        
    
    

         
            
        
        