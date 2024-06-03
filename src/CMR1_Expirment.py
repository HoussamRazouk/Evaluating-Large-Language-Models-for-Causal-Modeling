from openai import OpenAI
import sys
sys.path.append('.')
from src.init import init# just sets the API key as os variable 
from data.events import load_events
import json
import pandas as pd


models=["llama3-70b","mixtral-8x22b-instruct","gpt-3.5-turbo","gpt-4-turbo"]

