# Add your utilities or helper functions to this file.

import os
from dotenv import load_dotenv, find_dotenv

# these expect to find a .env file at the directory above the lesson.
# the format for that file is (without the comment)
# API_KEYNAME=AStringThatIsTheLongAPIKeyFromSomeService

def load_env():
    _ = load_dotenv(find_dotenv())

def get_serper_api_key():
    load_env()
    serper_api_key = os.getenv("SERPER_API_KEY")
    return serper_api_key

def get_openai_api_key():
    load_env()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return openai_api_key

def get_base_url():
    load_env()
    base_url = os.getenv("BASE_URL")
    return base_url

def get_trello_api_key():
    load_env()
    trello_api_key = os.getenv("TRELLO_API_KEY")
    return trello_api_key

def get_trello_token():
    load_env()
    trello_token = os.getenv("TRELLO_API_TOKEN")
    return trello_token

def get_trello_board_id():
    load_env()
    trello_board_id = os.getenv("TRELLO_BOARD_ID")
    return trello_board_id