from openai import OpenAI
from config import OPENAI_API_KEY

# Pass the API key directly when initializing the client
client = OpenAI(api_key=OPENAI_API_KEY)

client.models.list()
print(client.models.list())