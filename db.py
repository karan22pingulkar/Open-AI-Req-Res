import os
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables from .env
load_dotenv()

mongo_uri = os.environ.get("MONGO_URI")
db_name = os.environ.get("DB_NAME")

# Initialize client and select database
client = MongoClient(mongo_uri)
db = client[db_name]
