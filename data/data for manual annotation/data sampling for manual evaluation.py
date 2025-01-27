import pandas as pd
from sklearn.model_selection import train_test_split
file_name="results/CMR1/sampled_data\sampled_data_set_large.csv"
df=pd.read_csv(file_name)


X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

print(X_test)

X_test.to_csv('data/data for manual annotation/CMR1/TASK1_original.csv',index=False)


X_test[['Text1','Text2','domain']].to_csv('data/data for manual annotation/CMR1/TASK1_anonymized.csv',index=False)



import pandas as pd
from sklearn.model_selection import train_test_split
file_name="results/CMR2/sampled_data\sampled_data_set_large.csv"
df=pd.read_csv(file_name)


X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

print(X_test)

X_test.to_csv('data/data for manual annotation/CMR2/TASK2_original.csv',index=False)


X_test[['Value','Variable definition','domain']].to_csv('data/data for manual annotation/CMR2/TASK2_anonymized.csv',index=False)