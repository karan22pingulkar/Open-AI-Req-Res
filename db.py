import os
from dotenv import load_dotenv
from pymongo import MongoClient


load_dotenv()

mongo_uri = os.environ.get("MONGO_URI")
db_name = os.environ.get("DB_NAME")


client = MongoClient(mongo_uri)
db = client[db_name]
