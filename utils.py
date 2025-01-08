# utils.py

import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables from a .env file
def load_env():
    _ = load_dotenv(find_dotenv())

# Get the OpenAI API key from the environment variables
def get_openai_api_key():
    load_env()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OpenAI API key not found in environment variables.")
    return openai_api_key

# Get the Serper API key from the environment variables
def get_serper_api_key():
    load_env()
    serper_api_key = os.getenv("SERPER_API_KEY")
    if not serper_api_key:
        raise ValueError("Serper API key not found in environment variables.")
    return serper_api_key

# Break lines every 80 characters if the line is longer than 80 characters
# Don't break in the middle of a word
def pretty_print_result(result):
    parsed_result = []
    for line in result.split('\n'):
        if len(line) > 80:
            words = line.split(' ')
            new_line = ''
            for word in words:
                if len(new_line) + len(word) + 1 > 80:
                    parsed_result.append(new_line)
                    new_line = word
                else:
                    if new_line == '':
                        new_line = word
                    else:
                        new_line += ' ' + word
            parsed_result.append(new_line)
        else:
            parsed_result.append(line)
    return "\n".join(parsed_result)