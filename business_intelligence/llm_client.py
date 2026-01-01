import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# ------------------- LLM CLIENT -------------------

def get_llm_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

