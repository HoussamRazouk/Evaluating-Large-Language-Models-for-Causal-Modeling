from openai import OpenAI
import openai
import sys
sys.path.append('.')
from src.init import init
import pandas as pd
from src.CMR1.config import conf_init
from tqdm.auto import tqdm

from src.CMR1.get_cos_sim import get_cos_sim


tqdm.pandas()

config=conf_init()
models=config['models']
input_file=config['CMR1_sample_data_file']
output_path=config['CMR1_evaluated_data_dir']
#input_file='results/CMR1/test.csv'
test_data=pd.read_csv(input_file)

client = init()
#(text1,text2,model)
#Text1,Text2,Same Causal Variable,Variable Name,model Name,domain
embeddings_model='text-embedding-3-large'
test_data[f'text1_text2_cos_sim_{embeddings_model}']=test_data.progress_apply(lambda row: get_cos_sim(row['Text1'],row['Text2'],embeddings_model),axis=1)


test_data.to_csv(output_path+f'cos_sim_{embeddings_model}.csv',index=False)


