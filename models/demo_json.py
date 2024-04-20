from openai import OpenAI
import json
import os
from colorama import Fore, Style

client = OpenAI(api_key=os.getenv("OPENAI_API"))

def load_course_info(file_path):
    """
    Loads the course information from a JSON file.
    """
    with open(file_path, 'r') as file:
        course_info = json.load(file)
    return course_info

# for traditional NLP, please modify this function for your json data structure
def format_course_info_for_gpt(course_info):
    """
    Formats the course information into a text summary that GPT can understand.
    """
    summaries = []
    for course, details in course_info.items():
        concepts = ', '.join([concept['name'] for concept in details['Concepts']])
        summary = f"{details['Teaching Week']}: {details['Title']} covers {concepts}."
        summaries.append(summary)
    return " ".join(summaries)

def query_gpt_with_course_info(query, course_info_summary):
    """
    Submits a query along with the formatted course information to GPT for processing,
    using the newer client.chat.completions.create interface.
    """
    messages = [
        {"role": "system", "content": f"{course_info_summary} Your task is to provide a comprehensive response based on the information provided."},
        {"role": "user", "content": query}
    ]
    
    try:
        chat_completion = client.chat.completions.create(
            model="gpt-4-turbo",  # Adjust the model as needed
            messages=messages,
        )
        # Assuming we need to get the last message from the chat completion properly
        # Adjusting for the correct way to access the last message's content
        if chat_completion.choices and len(chat_completion.choices) > 0:
            last_message = chat_completion.choices[0].message.content.strip()
            return last_message
        else:
            return "No response was returned by GPT."
    except Exception as e:
        return f"Error querying GPT with course info: {e}"

file_path = "./data/json/gpt.json"
course_info = load_course_info(file_path)
course_info_summary = format_course_info_for_gpt(course_info)

print(Style.BRIGHT + Fore.BLUE + """
________                         _________________ ________
___  __ \______ ___________________  ____/___  __ \___  __/
__  /_/ /_  __ `/__  ___/__  ___/_  / __  __  /_/ /__  /   
_  ____/ / /_/ / _(__  ) _(__  ) / /_/ /  _  ____/ _  /    
/_/      \__,_/  /____/  /____/  \____/   /_/      /_/     
                                                           

""" + Style.RESET_ALL)

while True:
    query = input("Please enter your question: ")
    if query.lower() == 'exit':
        break
    response = query_gpt_with_course_info(query, course_info_summary)
    print(response)