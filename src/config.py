import os

from dotenv import load_dotenv


load_dotenv()


NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")

BASE_URL = "https://api.studio.nebius.ai/v1/"
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"