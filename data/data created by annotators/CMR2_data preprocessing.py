
import pandas as pd 
from ast import literal_eval ## list saved as strings 
from random import randrange

df_variables_values=pd.read_csv('data/data created by annotators/combined.csv')

df_variables_values['Values'] = df_variables_values['Values'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    
Selected_Causal_Variables = ['Coach', 'Team Motivation', 'Fitness Level', 'Warm Up Exercises', 'Genetics', 'Connective Tissue Disorder', 'Contact sport', 'Tissue Weakness', 'Neuromuscular Fatigue', 'Injury']
    
for index, row in df_variables_values.iterrows():
        if row['Causal Variable'] not in Selected_Causal_Variables:
            df_variables_values.drop(index, inplace=True)
df_variables_values= df_variables_values.reset_index(drop=True)


labels_list=[]
for labels in df_variables_values['Values']:
    
    labels_list=labels_list+labels
    
len(labels_list)
len(set(labels_list))

labels_set=list(set(labels_list))
list_variables=[]

for value in labels_set:
    
    variables=[]
    
    for _,row in df_variables_values.iterrows():
        
        if value in row['Values']:
            
            variables=variables+[row['Causal Variable']]
    
    list_variables.append(variables)

#Value,Variable definition,Interaction Value,model Name,domain

interaction_entity_df=pd.DataFrame(data={
                                    'Value': labels_set,
                                    'Variable definition': list_variables
                                })

interaction_entity_df['len']=interaction_entity_df.apply(lambda row: len(row['Variable definition']), axis=1)    
    

positive_interaction_entity_df=interaction_entity_df[interaction_entity_df['len']>1]

positive_interaction_entity_df['not in the variables']=positive_interaction_entity_df.apply(lambda row: row['Value'] not in row['Variable definition'], axis=1)

positive_interaction_entity_df=positive_interaction_entity_df[positive_interaction_entity_df['not in the variables']] 

positive_interaction_entity_df=positive_interaction_entity_df.reset_index(drop=True)

interaction_entity_df=pd.DataFrame(data={
                                    'Value': labels_set,
                                    'Variable definition': list_variables
                                })
Negative_samples=[]
for i in range(len(positive_interaction_entity_df)):
    
    while True:
        
        index1=randrange(len(interaction_entity_df))
        index2=randrange(len(interaction_entity_df))
        if index1!=index2:
            Value1=interaction_entity_df['Value'][index1]
            Value2=interaction_entity_df['Value'][index2]
            Variables1=interaction_entity_df['Variable definition'][index1]
            Variables2=interaction_entity_df['Variable definition'][index2]
            Sample_Variables=Variables1+Variables2
            Negative_samples.append(
                {
                   'Value':Value1,
                    'Variable definition':Sample_Variables
                }
            )
            Negative_samples.append(
                {
                   'Value':Value2,
                    'Variable definition':Sample_Variables
                }
            )
            break
        
negative_interaction_entity_df=pd.DataFrame(Negative_samples)       
 
positive_interaction_entity_df=positive_interaction_entity_df[['Value', 'Variable definition']]    
negative_interaction_entity_df['Interaction Value']=False
positive_interaction_entity_df['Interaction Value']=True    

positive_negative_interaction_entity_df=pd.concat([negative_interaction_entity_df, positive_interaction_entity_df], axis=0, ignore_index=True)
positive_negative_interaction_entity_df.to_csv('data/data created by annotators/CMR2_positive_negative_examples.csv', index=False)


