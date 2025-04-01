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
    Getting a wider prospective annotation if one of the annotators argued that the two text belong to the same causal variables then it is considered as true
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

def Getting_annotated_items_from_LLM_predictions(Annotation,LLM_prediction_path:str):
    '''
    Getting the data shown to  the annotators and predicted by the LLMs  back
    '''
    
    predictions_df=pd.read_csv(LLM_prediction_path)
    predictions_df = predictions_df.drop_duplicates(ignore_index=True)
    
    annotated_llm_predictions=[]
    
    for _,row in Annotation.iterrows():
        
        #Text1,Text2,Same Causal Variable,Variable Name,model Name,domain
        
        mask =  (predictions_df['Text1'] == row['Text1'] ) &\
                (predictions_df['Text2'] == row['Text2'] ) &\
                (predictions_df['Generated Same Causal Variable'] == row['Same Causal Variable']) &\
                (predictions_df['Data Generation Model'] == row['model Name'] ) &\
                (predictions_df['Domain'] == row['domain'])
        
        assert(len(predictions_df[mask])==1)
        
        for _,row in predictions_df[mask].iterrows():
            annotated_llm_prediction_dict={}
            annotated_llm_prediction_dict['Text1']=row['Text1']
            annotated_llm_prediction_dict['Text2']=row['Text2']
            annotated_llm_prediction_dict['Same Causal Variable']=row['Predicted Same Causal Variable']
            annotated_llm_prediction_dict['Variable Name']=row['Predicted Variable Name']
            annotated_llm_prediction_dict['model Name']=row['Data Generation Model']
            annotated_llm_prediction_dict['domain']=row['Domain']
            annotated_llm_predictions.append(annotated_llm_prediction_dict)
    
    
    return pd.DataFrame(annotated_llm_predictions)         
        
## reading data from files
Anno1_df=pd.read_excel("data/data for manual annotation/Annotated data/CMR1_Annotator1.xlsx")
Anno2_df=pd.read_excel("data/data for manual annotation/Annotated data/CMR1_Annotator2.xlsx")
Anno3_df=pd.read_excel("data/data for manual annotation/Annotated data/CMR1_Annotator3.xlsx")
Generated_Data_df=pd.read_csv("data/data for manual annotation/CMR1/TASK1_original.csv")


## Printing the unique values of the each of data set
print(Anno1_df['Same Causal Variable'].unique())
print(Anno2_df['Same Causal Variable'].unique())
print(Anno3_df['Same Causal Variable'].unique())
print(Generated_Data_df['Same Causal Variable'].unique())


## Post processing the annotated data set for different format to True False or not annotated

Anno1_df['Same Causal Variable_processed']=Post_Pro_Annotation(Anno1_df['Same Causal Variable'],[True], [False])

Anno2_df['Same Causal Variable_processed']=Post_Pro_Annotation(Anno2_df['Same Causal Variable'],['Yes','yes','Yes '], ['No'])

Anno3_df['Same Causal Variable_processed']=Post_Pro_Annotation(Anno3_df['Same Causal Variable'],[1], [0])
# not needed for Generated_Data_df


## Calculating the inter annotator agreement between the annotators and each other 

filtered_DS1,filtered_DS2=getting_the_overlapped_between_two_annotated_data_set(
    Anno1_df['Same Causal Variable_processed'],
    Anno2_df['Same Causal Variable_processed']
)

IAA_anno1_anno2=round(cohen_kappa_score(filtered_DS1,filtered_DS2),2)*100


filtered_DS1,filtered_DS2=getting_the_overlapped_between_two_annotated_data_set(
    Anno2_df['Same Causal Variable_processed'],
    Anno3_df['Same Causal Variable_processed']
)

IAA_anno2_anno3=round(cohen_kappa_score(filtered_DS1,filtered_DS2),2)*100   


filtered_DS1,filtered_DS2=getting_the_overlapped_between_two_annotated_data_set(
    Anno3_df['Same Causal Variable_processed'],
    Anno1_df['Same Causal Variable_processed']
)
IAA_anno3_anno1=round(cohen_kappa_score(filtered_DS1,filtered_DS2),2)*100

avg_IAA_anno_anno=round((IAA_anno3_anno1+IAA_anno1_anno2+IAA_anno2_anno3)/3,0)
print(f"IAA ANNOTATOR 1 & ANNOTATOR 2 ={IAA_anno1_anno2}")
print(f"IAA ANNOTATOR 2 & ANNOTATOR 3 ={IAA_anno2_anno3}")
print(f"IAA ANNOTATOR 3 & ANNOTATOR 1 ={IAA_anno3_anno1}")
print(f"IAA ANNOTATOR  & ANNOTATOR AVG ={avg_IAA_anno_anno}")



## Calculating the inter annotator agreement between the annotators and the generated dataset 

WN_index=45        #Number of data points from Word net


filtered_DS1,filtered_DS2=getting_the_overlapped_between_two_annotated_data_set(
    Anno1_df['Same Causal Variable_processed'][WN_index:].reset_index(drop=True),
    Generated_Data_df['Same Causal Variable']
)

IAA_anno1_GD=round(cohen_kappa_score(filtered_DS1,filtered_DS2),2)*100


filtered_DS1,filtered_DS2=getting_the_overlapped_between_two_annotated_data_set(
    Anno2_df['Same Causal Variable_processed'][WN_index:].reset_index(drop=True),
    Generated_Data_df['Same Causal Variable']
)

IAA_anno2_GD=round(cohen_kappa_score(filtered_DS1,filtered_DS2),2)*100


filtered_DS1,filtered_DS2=getting_the_overlapped_between_two_annotated_data_set(
    Anno3_df['Same Causal Variable_processed'][WN_index:].reset_index(drop=True),
    Generated_Data_df['Same Causal Variable']
)

IAA_anno3_GD=round(cohen_kappa_score(filtered_DS1,filtered_DS2),2)*100 
avg_IAA_anno_GD=round((IAA_anno1_GD+IAA_anno2_GD+IAA_anno3_GD)/3,0)
print(f"IAA ANNOTATOR 1 & Generated Data = {IAA_anno1_GD}")
print(f"IAA ANNOTATOR 2 & Generated Data = {IAA_anno2_GD}")
print(f"IAA ANNOTATOR 3 & Generated Data = {IAA_anno3_GD}")
print(f"IAA ANNOTATOR  & Generated Data avg = {avg_IAA_anno_GD}")



unified_annotation=Getting_the_union_of_the_three_annotators(
    Anno1_df['Same Causal Variable_processed'],
    Anno2_df['Same Causal Variable_processed'],
    Anno3_df['Same Causal Variable_processed'],
)

filtered_DS1,filtered_DS2=getting_the_overlapped_between_two_annotated_data_set(
    unified_annotation[WN_index:],
    Generated_Data_df['Same Causal Variable']
)
IAA_annoUN_GD=round(cohen_kappa_score(filtered_DS1,filtered_DS2),2)*100
print(f"IAA ANNOTATORs Unified & Generated Data = {IAA_annoUN_GD}")


### count the agreement if thy hyponyms relations in word net represent a relation between a value and a causal variable
Anno1_df['Same Causal Variable_processed'][:WN_index]
Anno2_df['Same Causal Variable_processed'][:WN_index]
Anno3_df['Same Causal Variable_processed'][:WN_index]

unified_annotation[:WN_index]


Anno1_WN_Trues=sum(1 for value in Anno1_df['Same Causal Variable_processed'][:WN_index] if value==True)
Anno2_WN_Trues=sum(1 for value in Anno2_df['Same Causal Variable_processed'][:WN_index] if value==True)
Anno3_WN_Trues=sum(1 for value in Anno3_df['Same Causal Variable_processed'][:WN_index] if value==True)
unified_annotation_WN_Trues=sum(1 for value in unified_annotation[:WN_index] if value==True)

Anno1_WN_False=sum(1 for value in Anno1_df['Same Causal Variable_processed'][:WN_index] if (value==False))
Anno2_WN_False=sum(1 for value in Anno2_df['Same Causal Variable_processed'][:WN_index] if (value==False))
Anno3_WN_False=sum(1 for value in Anno3_df['Same Causal Variable_processed'][:WN_index] if (value==False))
unified_annotation_WN_False=sum(1 for value in unified_annotation[:WN_index] if (value==False))

AA_anno1_WN=Anno1_WN_Trues/(Anno1_WN_Trues+Anno1_WN_False)
AA_anno2_WN=Anno2_WN_Trues/(Anno2_WN_Trues+Anno2_WN_False)
AA_anno3_WN=Anno3_WN_Trues/(Anno3_WN_Trues+Anno3_WN_False)
AA_unified_annotation_WN=unified_annotation_WN_Trues/(unified_annotation_WN_Trues+unified_annotation_WN_False)


avg_AA_anno_WN=round((AA_anno1_WN+AA_anno2_WN+AA_anno3_WN)/3,2)*100

print(f"AA ANNOTATORs 1 & Word net = {AA_anno1_WN}")
print(f"AA ANNOTATORs 2 & Word net = {AA_anno2_WN}")
print(f"AA ANNOTATORs 3 & Word net = {AA_anno3_WN}")
print(f"AA ANNOTATORs avg & Word net = {avg_AA_anno_WN}")

print(f"AA Unified ANNOTATORs & Word net = {AA_unified_annotation_WN}")

## getting the data predicted by llama3_70B and comparing to the different annotators 


LLM_prediction_path='results/CMR1/evaluated_data/llama3-70b_model_prediction_large.csv'

lama3_70B_predictions=Getting_annotated_items_from_LLM_predictions(Generated_Data_df,LLM_prediction_path)


filtered_DS1,filtered_DS2=getting_the_overlapped_between_two_annotated_data_set(
    Anno1_df['Same Causal Variable_processed'][WN_index:].reset_index(drop=True),
    lama3_70B_predictions['Same Causal Variable']
)

IAA_anno1_lama3_70B=round(cohen_kappa_score(filtered_DS1,filtered_DS2),2)*100


filtered_DS1,filtered_DS2=getting_the_overlapped_between_two_annotated_data_set(
    Anno2_df['Same Causal Variable_processed'][WN_index:].reset_index(drop=True),
    lama3_70B_predictions['Same Causal Variable']
)

IAA_anno2_lama3_70B=round(cohen_kappa_score(filtered_DS1,filtered_DS2),2)*100


filtered_DS1,filtered_DS2=getting_the_overlapped_between_two_annotated_data_set(
    Anno3_df['Same Causal Variable_processed'][WN_index:].reset_index(drop=True),
    lama3_70B_predictions['Same Causal Variable']
)

IAA_anno3_lama3_70B=round(cohen_kappa_score(filtered_DS1,filtered_DS2),2)*100 
avg_IAA_anno_lama3_70B=round((IAA_anno1_lama3_70B+IAA_anno2_lama3_70B+IAA_anno3_lama3_70B)/3,0)

print(f"IAA ANNOTATOR 1 & Predicted data by  lama3_70B = {IAA_anno1_lama3_70B}")
print(f"IAA ANNOTATOR 2 & Predicted data by  lama3_70B = {IAA_anno2_lama3_70B}")
print(f"IAA ANNOTATOR 3 & Predicted data by  lama3_70B = {IAA_anno3_lama3_70B}")
print(f"IAA ANNOTATOR avg & Predicted data by  lama3_70B = {avg_IAA_anno_lama3_70B}")


## getting the data predicted by gpt-4-turbo and comparing to the different annotators 


    
#'results/CMR1/evaluated_data/gpt-4-turbo_model_prediction_large.csv'


LLM_prediction_path='results/CMR1/evaluated_data/gpt-4-turbo_model_prediction_large.csv'

gpt_4_turbo_predictions=Getting_annotated_items_from_LLM_predictions(Generated_Data_df,LLM_prediction_path)


filtered_DS1,filtered_DS2=getting_the_overlapped_between_two_annotated_data_set(
    Anno1_df['Same Causal Variable_processed'][WN_index:].reset_index(drop=True),
    gpt_4_turbo_predictions['Same Causal Variable']
)

IAA_anno1_gpt_4_turbo=round(cohen_kappa_score(filtered_DS1,filtered_DS2),2)*100


filtered_DS1,filtered_DS2=getting_the_overlapped_between_two_annotated_data_set(
    Anno2_df['Same Causal Variable_processed'][WN_index:].reset_index(drop=True),
    gpt_4_turbo_predictions['Same Causal Variable']
)

IAA_anno2_gpt_4_turbo=round(cohen_kappa_score(filtered_DS1,filtered_DS2),2)*100


filtered_DS1,filtered_DS2=getting_the_overlapped_between_two_annotated_data_set(
    Anno3_df['Same Causal Variable_processed'][WN_index:].reset_index(drop=True),
    gpt_4_turbo_predictions['Same Causal Variable']
)

IAA_anno3_gpt_4_turbo=round(cohen_kappa_score(filtered_DS1,filtered_DS2),2)*100 
avg_IAA_anno_gpt_4_turbo=round((IAA_anno1_gpt_4_turbo+IAA_anno2_gpt_4_turbo+IAA_anno3_gpt_4_turbo)/3,0)
print(f"IAA ANNOTATOR 1 & Predicted data by  gpt_4_turbo = {IAA_anno1_gpt_4_turbo}")
print(f"IAA ANNOTATOR 2 & Predicted data by  gpt_4_turbo = {IAA_anno2_gpt_4_turbo}")
print(f"IAA ANNOTATOR 3 & Predicted data by  gpt_4_turbo = {IAA_anno3_gpt_4_turbo}")
print(f"IAA ANNOTATOR avg & Predicted data by  gpt_4_turbo = {avg_IAA_anno_gpt_4_turbo}")




## getting the data predicted by llama3-8b and comparing to the different annotators 




'results/CMR1/evaluated_data/llama3-8b_model_prediction_large.csv'


LLM_prediction_path='results/CMR1/evaluated_data/llama3-8b_model_prediction_large.csv'

llama3_8b_predictions=Getting_annotated_items_from_LLM_predictions(Generated_Data_df,LLM_prediction_path)


filtered_DS1,filtered_DS2=getting_the_overlapped_between_two_annotated_data_set(
    Anno1_df['Same Causal Variable_processed'][WN_index:].reset_index(drop=True),
    llama3_8b_predictions['Same Causal Variable']
)

IAA_anno1_llama3_8b=round(cohen_kappa_score(filtered_DS1,filtered_DS2),2)*100


filtered_DS1,filtered_DS2=getting_the_overlapped_between_two_annotated_data_set(
    Anno2_df['Same Causal Variable_processed'][WN_index:].reset_index(drop=True),
    llama3_8b_predictions['Same Causal Variable']
)

IAA_anno2_llama3_8b=round(cohen_kappa_score(filtered_DS1,filtered_DS2),2)*100


filtered_DS1,filtered_DS2=getting_the_overlapped_between_two_annotated_data_set(
    Anno3_df['Same Causal Variable_processed'][WN_index:].reset_index(drop=True),
    llama3_8b_predictions['Same Causal Variable']
)

IAA_anno3_llama3_8b=round(cohen_kappa_score(filtered_DS1,filtered_DS2),2)*100 
avg_IAA_anno_llama3_8b=round((IAA_anno1_llama3_8b+IAA_anno2_llama3_8b+IAA_anno3_llama3_8b)/3,0)
print(f"IAA ANNOTATOR 1 & Predicted data by  llama3_8b = {IAA_anno1_llama3_8b}")
print(f"IAA ANNOTATOR 2 & Predicted data by  llama3_8b = {IAA_anno2_llama3_8b}")
print(f"IAA ANNOTATOR 3 & Predicted data by  llama3_8b = {IAA_anno3_llama3_8b}")
print(f"IAA ANNOTATOR avg & Predicted data by  llama3_8b = {avg_IAA_anno_llama3_8b}")




LLM_prediction_path='results/CMR1/evaluated_data/mixtral-8x22b-instruct_model_prediction_large.csv'

mixtral_8x22b_predictions=Getting_annotated_items_from_LLM_predictions(Generated_Data_df,LLM_prediction_path)


filtered_DS1,filtered_DS2=getting_the_overlapped_between_two_annotated_data_set(
    Anno1_df['Same Causal Variable_processed'][WN_index:].reset_index(drop=True),
    mixtral_8x22b_predictions['Same Causal Variable']
)

IAA_anno1_mixtral_8x22b=round(cohen_kappa_score(filtered_DS1,filtered_DS2),2)*100


filtered_DS1,filtered_DS2=getting_the_overlapped_between_two_annotated_data_set(
    Anno2_df['Same Causal Variable_processed'][WN_index:].reset_index(drop=True),
    mixtral_8x22b_predictions['Same Causal Variable']
)

IAA_anno2_mixtral_8x22b=round(cohen_kappa_score(filtered_DS1,filtered_DS2),2)*100


filtered_DS1,filtered_DS2=getting_the_overlapped_between_two_annotated_data_set(
    Anno3_df['Same Causal Variable_processed'][WN_index:].reset_index(drop=True),
    mixtral_8x22b_predictions['Same Causal Variable']
)

IAA_anno3_mixtral_8x22b=round(cohen_kappa_score(filtered_DS1,filtered_DS2),2)*100 
avg_IAA_anno_mixtral_8x22b=round((IAA_anno1_mixtral_8x22b+IAA_anno2_mixtral_8x22b+IAA_anno3_mixtral_8x22b)/3,0)
print(f"IAA ANNOTATOR 1 & Predicted data by  mixtral_8x22b = {IAA_anno1_mixtral_8x22b}")
print(f"IAA ANNOTATOR 2 & Predicted data by  mixtral_8x22b = {IAA_anno2_mixtral_8x22b}")
print(f"IAA ANNOTATOR 3 & Predicted data by  mixtral_8x22b = {IAA_anno3_mixtral_8x22b}")
print(f"IAA ANNOTATOR avg & Predicted data by  mixtral_8x22b = {avg_IAA_anno_mixtral_8x22b}")



