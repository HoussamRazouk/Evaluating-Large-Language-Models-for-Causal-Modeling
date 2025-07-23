

import pandas as pd

import itertools

import editdistance
import random
from random import randrange
from tqdm.auto import tqdm
tqdm.pandas()


def get_concepts_and_labels(file_path):
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
    distance_threshold = 2
    for i in range(len(df_label_labels['labels'])):
        test=df_label_labels['labels'][i]
        edited=True
        while edited:
            edited=False
            combinations=list(itertools.combinations(test,2))
            for combination in combinations:
                distance=editdistance.eval(combination[0],combination[1])
                if distance<distance_threshold: 
                    test.remove(combination[1])
                    print(combination[1])
                    print(combination)
                    edited=True
                    break 

    return df_label_labels




def get_positive_and_negative_samples(df_label_labels, number_of_samples=3):
    positive_samples=[]
    
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
                    number_of_appended_samples+=1
                    if number_of_appended_samples==number_of_samples:
                        break

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
            
            if sample in negative_samples or sample_rev in negative_samples:
                continue
            else:
                negative_samples.append(sample)
                number_of_appended_negative_samples+=1
                if number_of_appended_negative_samples==len(positive_samples):
                    break

    return positive_samples, negative_samples

def process_samples_to_df(positive_samples, negative_samples):
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
    return sampled_data_df
    
            
            
def main():
    file_path = "data/wikicausal/enwiki-20220901-kg_v6-corpus_v4.jsonl"
    df_label_labels = get_concepts_and_labels(file_path)
    
    positive_samples, negative_samples = get_positive_and_negative_samples(df_label_labels)
    
    sampled_data_df = process_samples_to_df(positive_samples, negative_samples)
    
    sampled_data_df.to_csv('data/wikicausal/CMR1_positive_negative_examples_enwiki-20220901-kg_v6-corpus_v4.jsonl.csv', index=False)
    
