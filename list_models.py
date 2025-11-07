# list_models.py
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise SystemExit("GOOGLE_API_KEY not set in .env")

genai.configure(api_key=api_key)

# List models available to this client / API version
models = genai.list_models()  # returns a list of model dicts
print("Available model IDs (first 50):")
for m in models[:50]:
    # model objects may have 'name' or 'id' depending on library version
    print(m.get("name") or m.get("id") or m)
