import nltk
from nltk.corpus import wordnet
import pandas as pd 
import random


'''
Checking the generated data set for hyponyms.
We assume that the hyponyms of a text entity representing a causal variable could represent different values of that causal variables
'''

# Initialize the NLTK WordNet interface
nltk.download('wordnet')

df=pd.read_csv("results\CMR1\sampled_data\sampled_data_set_large.csv")
i=0
df_list=list(set(df['Variable Name'].tolist()))



len(df_list)-1
Dictionaries=[]
for val in df_list:
    
    if type(val)==str:
        synsets = wordnet.synsets(val)
        if synsets!=[]:
            #print(synsets)
            for  syn in synsets:
                    row={}
                    if syn.hyponyms()!= []:
                        if '.n.' in syn.name():# only noun phrases are considered 
                            if len(syn.hyponyms())>1:# only values with two or  more  hyponyms
                                domain=list(set(df[df['Variable Name']==val]['domain'].tolist()))
                                print("Variable:", val)
                                row["Variable"]=val
                                print("Domain:", domain)
                                row["Domain"]=str(domain)[2:-2]
                                print("Word:", syn.name())
                                row["Word"]=syn.name()
                                print("Definition:", syn.definition())
                                row["Definition"]=syn.definition()
                                hyponyms=[hypo.name() for hypo in syn.hyponyms()]
                                print("hyponyms:", [hypo.name() for hypo in syn.hyponyms()])
                                i=i+1
                                print(i)
                                counter=0
                                Examples=[]
                                while True:
                                    
                                    random_number = random.randint(0, len(hyponyms)-1) 
                                    if hyponyms[random_number].capitalize().split('.n.')[0] not in Examples:
                                        Examples.append(hyponyms[random_number].capitalize().split('.n.')[0] )
                                        counter=counter+1
                                    if counter==2:
                                        break
                                
                                print("Text1:", Examples[0])
                                row["Text1"]= Examples[0]
                                print("Text2:", Examples[1])
                                row["Text2"]=  Examples[1] 
                                Dictionaries.append(row)
                                #break ## limited number of variables are found in word_net! 14 out 339
                                

results=pd.DataFrame(Dictionaries)                    
print(results)
'''Text1,Text2,Same Causal Variable,Variable Name,model Name,domain'''
results=results[['Text1','Text2','Variable','Word','Definition','Domain']]       
results.to_csv('data/data for manual annotation/CMR1_word_net/TASK1_original.csv',index=False)
results[['Text1','Text2','Domain']].to_csv('data/data for manual annotation/CMR1_word_net/TASK1_anonymized.csv',index=False)
