from openai import OpenAI
import openai
import sys
sys.path.append('.')
from src.init import init
from sklearn.metrics.pairwise import cosine_similarity

client = init
text1="High-Crime Area"
text2="Disaster-Prone Region"
model="text-embedding-3-small"
openai.Embedding.create
client.
emb_text1=openai.Embedding.create(input = [text1], model=model).data[0]
emb_text2=client.Embeddings.create(input = [text2], model=model).data[0]

cos_sim = cosine_similarity(emb_text1, emb_text2)