import pandas as pd
from sklearn.metrics import cohen_kappa_score
import itertools
"""
df=pd.read_excel("data/data for manual annotation/Annotated data/CMR1_Annotator1.xlsx")
df.to_csv("data/data for manual annotation/Annotated data/CMR1_Annotator1.csv",index=False)
df=pd.read_csv("data/data for manual annotation/Annotated data/CMR1_Annotator1.csv")
df.to_excel("data/data for manual annotation/Annotated data/CMR1_Annotator1_corrected.xlsx",index=False)
"""

## Post processing the annotated data set for different format to True False or not annotated
def Post_Pro_Annotation(df_annotation,True_annotation_values, False_annotation_values,no_annotation_index='Not_annotated'):
    Processed_annotation=[]
    for annotation in df_annotation:
        if annotation in True_annotation_values:
            
            Processed_annotation.append(True)
        
        elif annotation in False_annotation_values:
            
            Processed_annotation.append(False)
        
        else:
            
            Processed_annotation.append(no_annotation_index)
    
    return Processed_annotation

def getting_the_overlapped_between_two_annotated_data_set(data_set1:list,data_set2:list,no_annotation_index='Not_annotated'):
    '''
    For removing data which is not annotated by both annotators
    
    '''
    filtered_DS1=[]
    filtered_DS2=[]
    
    assert(len(data_set1)==len(data_set2))
    for i in range(len(data_set1)):
        
        if (data_set1[i]==no_annotation_index or data_set2[i]==no_annotation_index):
        
            continue
        
        else:
            
            filtered_DS1.append(data_set1[i])
            filtered_DS2.append(data_set2[i])
    
    return filtered_DS1,filtered_DS2
            
def Getting_the_union_of_the_three_annotators(df_annotation1,df_annotation2,df_annotation3,no_annotation_index='Not_annotated'):
    '''
    Getting a wider prospective annotation if one of the annotators argued that the two text belong to the Interaction Values then it is considered as true
    '''
    
    assert(len(df_annotation1)==len(df_annotation2)==len(df_annotation3))
    
    unified_annotation=[]
    
    for i in range(len(df_annotation1)):
    
        if True in [df_annotation1[i],df_annotation2[i],df_annotation3[i]]:
            
            unified_annotation.append(True)
        
        elif False in [df_annotation1[i],df_annotation2[i],df_annotation3[i]]:
            
            unified_annotation.append(False)
            
        else :
            unified_annotation.append(no_annotation_index)
    
    return unified_annotation
            
        
## reading data from files
Anno1_df=pd.read_excel("data/data for manual annotation/Annotated data/CMR2_Annotator1.xlsx")
Anno2_df=pd.read_excel("data/data for manual annotation/Annotated data/CMR2_Annotator2.xlsx")
Anno3_df=pd.read_excel("data/data for manual annotation/Annotated data/CMR2_Annotator3.xlsx")
Generated_Data_df=pd.read_csv("data/data for manual annotation/CMR2/TASK2_original.csv")


## Printing the unique values of the each of data set
print(Anno1_df['Interaction Value'].unique())
print(Anno2_df['Interaction Value'].unique())
print(Anno3_df['Interaction Value'].unique())
print(Generated_Data_df['Interaction Value'].unique())


## Post processing the annotated data set for different format to True False or not annotated

Anno1_df['Interaction Value_processed']=Post_Pro_Annotation(Anno1_df['Interaction Value'],[1], [0])

Anno2_df['Interaction Value_processed']=Post_Pro_Annotation(Anno2_df['Interaction Value'],['Yes','yes?','Yes ','yes'], ['no'])

Anno3_df['Interaction Value_processed']=Post_Pro_Annotation(Anno3_df['Interaction Value'],[1], [0])
# not needed for Generated_Data_df


## Calculating the inter annotator agreement between the annotators and each other 

filtered_DS1,filtered_DS2=getting_the_overlapped_between_two_annotated_data_set(
    Anno1_df['Interaction Value_processed'],
    Anno2_df['Interaction Value_processed']
)

IAA_anno1_anno2=cohen_kappa_score(filtered_DS1,filtered_DS2)


filtered_DS1,filtered_DS2=getting_the_overlapped_between_two_annotated_data_set(
    Anno2_df['Interaction Value_processed'],
    Anno3_df['Interaction Value_processed']
)

IAA_anno2_anno3=cohen_kappa_score(filtered_DS1,filtered_DS2)   


filtered_DS1,filtered_DS2=getting_the_overlapped_between_two_annotated_data_set(
    Anno3_df['Interaction Value_processed'],
    Anno1_df['Interaction Value_processed']
)
IAA_anno3_anno1=cohen_kappa_score(filtered_DS1,filtered_DS2) 


print(f"IAA ANNOTATOR 1 & ANNOTATOR 2 ={IAA_anno1_anno2}")
print(f"IAA ANNOTATOR 2 & ANNOTATOR 3 ={IAA_anno2_anno3}")
print(f"IAA ANNOTATOR 3 & ANNOTATOR 1 ={IAA_anno3_anno1}")


## Calculating the inter annotator agreement between the annotators and the generated dataset 

WN_index=4        #Number of data points from Word net


filtered_DS1,filtered_DS2=getting_the_overlapped_between_two_annotated_data_set(
    Anno1_df['Interaction Value_processed'][WN_index:].reset_index(drop=True),
    Generated_Data_df['Interaction Value']
)

IAA_anno1_GD=cohen_kappa_score(filtered_DS1,filtered_DS2)


filtered_DS1,filtered_DS2=getting_the_overlapped_between_two_annotated_data_set(
    Anno2_df['Interaction Value_processed'][WN_index:].reset_index(drop=True),
    Generated_Data_df['Interaction Value']
)

IAA_anno2_GD=cohen_kappa_score(filtered_DS1,filtered_DS2)   


filtered_DS1,filtered_DS2=getting_the_overlapped_between_two_annotated_data_set(
    Anno3_df['Interaction Value_processed'][WN_index:].reset_index(drop=True),
    Generated_Data_df['Interaction Value']
)

IAA_anno3_GD=cohen_kappa_score(filtered_DS1,filtered_DS2) 

print(f"IAA ANNOTATOR 1 & Generated Data = {IAA_anno1_GD}")
print(f"IAA ANNOTATOR 2 & Generated Data = {IAA_anno2_GD}")
print(f"IAA ANNOTATOR 3 & Generated Data = {IAA_anno3_GD}")



unified_annotation=Getting_the_union_of_the_three_annotators(
    Anno1_df['Interaction Value_processed'],
    Anno2_df['Interaction Value_processed'],
    Anno3_df['Interaction Value_processed'],
)

filtered_DS1,filtered_DS2=getting_the_overlapped_between_two_annotated_data_set(
    unified_annotation[WN_index:],
    Generated_Data_df['Interaction Value']
)
IAA_annoUN_GD=cohen_kappa_score(filtered_DS1,filtered_DS2)
print(f"IAA ANNOTATORs Unified & Generated Data = {IAA_annoUN_GD}")


### count the agreement if thy hyponyms relations in word net represent a relation between a value and a causal variable
Anno1_df['Interaction Value_processed'][:WN_index]
Anno2_df['Interaction Value_processed'][:WN_index]
Anno3_df['Interaction Value_processed'][:WN_index]

unified_annotation[:WN_index]


Anno1_WN_Trues=sum(1 for value in Anno1_df['Interaction Value_processed'][:WN_index] if value==True)
Anno2_WN_Trues=sum(1 for value in Anno2_df['Interaction Value_processed'][:WN_index] if value==True)
Anno3_WN_Trues=sum(1 for value in Anno3_df['Interaction Value_processed'][:WN_index] if value==True)
unified_annotation_WN_Trues=sum(1 for value in unified_annotation[:WN_index] if value==True)

Anno1_WN_False=sum(1 for value in Anno1_df['Interaction Value_processed'][:WN_index] if (value==False))
Anno2_WN_False=sum(1 for value in Anno2_df['Interaction Value_processed'][:WN_index] if (value==False))
Anno3_WN_False=sum(1 for value in Anno3_df['Interaction Value_processed'][:WN_index] if (value==False))
unified_annotation_WN_False=sum(1 for value in unified_annotation[:WN_index] if (value==False))

AA_anno1_WN=Anno1_WN_Trues/(Anno1_WN_Trues+Anno1_WN_False)
AA_anno2_WN=Anno2_WN_Trues/(Anno2_WN_Trues+Anno2_WN_False)
AA_anno3_WN=Anno3_WN_Trues/(Anno3_WN_Trues+Anno3_WN_False)
AA_unified_annotation_WN=unified_annotation_WN_Trues/(unified_annotation_WN_Trues+unified_annotation_WN_False)

print(f"AA ANNOTATORs 1 & Word net = {AA_anno1_WN}")
print(f"AA ANNOTATORs 2 & Word net = {AA_anno2_WN}")
print(f"AA ANNOTATORs 3 & Word net = {AA_anno3_WN}")
print(f"AA Unified ANNOTATORs & Word net = {AA_unified_annotation_WN}")


