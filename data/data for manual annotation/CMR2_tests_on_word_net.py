import nltk
from nltk.corpus import wordnet
import pandas as pd 
import random


'''

Checking the generated data set for hypernyms.
We assume that the hypernyms of a text entity representing a values could represent the different values of different causal variable 

'''

# Initialize the NLTK WordNet interface
nltk.download('wordnet')

df=pd.read_csv("results\CMR2\sampled_data\sampled_data_set_large.csv")
i=0
df_list=list(set(df['Value'].tolist()))


len(df_list)-1
Dictionaries=[]
counter_of_values_found_in_wordnet=0
Dictionaries=[]
for val in df_list:
    
    if type(val)==str:
        synsets = wordnet.synsets(val)
        if synsets!=[]:
            counter_of_values_found_in_wordnet=counter_of_values_found_in_wordnet+1
            print(synsets)
            hypernyms=[]
            for  syn in synsets:
                if '.n.' in syn.name():
                    if syn.hypernyms()!=[]:
                        for hyper in syn.hypernyms():
                            if '.n.' in hyper.name():
                                hypernyms.append(hyper.name())
            if len(hypernyms)>1:
                row={}
                counter=0
                Examples=[]
                while True:
                                        
                    random_number = random.randint(0, len(hypernyms)-1) 
                    if hypernyms[random_number].capitalize().split('.n.')[0] not in Examples:
                        Examples.append(hypernyms[random_number].capitalize().split('.n.')[0] )
                        counter=counter+1
                    if counter==2:
                        break
                domain=list(set(df[df['Value']==val]['domain'].tolist()))            
                row["Domain"]=str(domain)[2:-2].capitalize()
                row['Variable definition']=str(Examples)    
                row["Value"]=val
                Dictionaries.append(row)
                
counter_of_values_found_in_wordnet
if False:            
    len(df_list)-1
    Dictionaries=[]
    for val in df_list:
        
        if type(val)==str:
            synsets = wordnet.synsets(val)
            if synsets!=[]:
                print(synsets)
                for  syn in synsets:
                        row={}
                        if syn.hypernyms()!= []:
                            if '.n.' in syn.name():# only noun phrases are considered 
                                if len(syn.hypernyms())>1:# only values with two or  more  hypernyms
                                    domain=list(set(df[df['Value']==val]['domain'].tolist()))
                                    print("Variable:", val)
                                    row["Value"]=val
                                    print("Domain:", domain)
                                    row["Domain"]=str(domain)[2:-2].capitalize()
                                    print("Word:", syn.name())
                                    row["Word"]=syn.name()
                                    print("Definition:", syn.definition())
                                    row["Definition"]=syn.definition()
                                    hypernyms=[hyper.name() for hyper in syn.hypernyms()]
                                    print("hypernyms:", [hyper.name() for hyper in syn.hypernyms()])
                                    i=i+1
                                    print(i)
                                    counter=0
                                    Examples=[]
                                    while True:
                                        
                                        random_number = random.randint(0, len(hypernyms)-1) 
                                        if hypernyms[random_number].capitalize().split('.n.')[0] not in Examples:
                                            Examples.append(hypernyms[random_number].capitalize().split('.n.')[0] )
                                            counter=counter+1
                                        if counter==2:
                                            break
                                    row['Variable definition']=str(Examples)
                                    Dictionaries.append(row)
                                    #break ## limited number of variables are found in word_net! 14 out 339
                                

results=pd.DataFrame(Dictionaries)                    
print(results)
'''Value,Variable definition,Interaction Value,model Name,domain'''
if len(results)>0:
    results=results[['Value','Variable definition','Domain']]       
    #results.to_csv('data/data for manual annotation/CMR2_word_net/TASK2_original.csv',index=False)
    #results[['Value','Variable definition','Domain']].to_csv('data/data for manual annotation/CMR2_word_net/TASK2_anonymized.csv',index=False)
