import os
from langchain_groq import ChatGroq

groq_api_key=os.environ['GROQ_API_KEY']

llm = ChatGroq(groq_api_key=groq_api_key,
               model_name = "mixtral-8x7b-32768")

