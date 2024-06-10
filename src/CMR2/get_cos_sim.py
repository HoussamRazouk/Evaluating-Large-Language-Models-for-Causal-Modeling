import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_cos_sim(text1,text2,model):
    emb_text1=openai.embeddings.create(input = [text1], model=model).data[0].embedding
    emb_text2=openai.embeddings.create(input = [text2], model=model).data[0].embedding

    cos_sim = cosine_similarity(np.array(emb_text1).reshape(1,-1) , np.array(emb_text2).reshape(1,-1) )

    return round(cos_sim[0][0],3)


if False:
    import sys
    sys.path.append('.')
    from src.init import init
    client=init()
    text1='High-Crime Area'
    text2='Disaster-Prone Region'
    model='gpt-3.5-turbo'
    emb_text1=openai.embeddings.create(input = [text1], model=model).data[0].embedding