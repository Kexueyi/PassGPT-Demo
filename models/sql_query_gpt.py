import openai as OpenAI
from app.database import get_db_connection
import os

# 初始化OpenAI客户端
client = OpenAI(api_key=os.getenv("OPENAI_API"))

def generate_database_query(user_input):
    """使用GPT生成数据库查询指令"""
    messages = [
        {"role": "system", "content": f"Generate a SQL query from passgpt database to fetch all related information about:, struct."},
        {"role": "user", "content": query}
    ]
    
    try:
        chat_completion = client.chat.completions.create(
            model="gpt-4-turbo",  # Adjust the model as needed
            messages=messages,
        )
        query = response.choices[0].text.strip()
        return query
    except Exception as e:
        print(f"Error generating query: {e}")
        return None

def execute_query(sql_query):
    """执行SQL查询并返回结果"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(sql_query)
        result = cursor.fetchall()
        return result
    except Exception as e:
        print(f"Error executing query: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

def sql_query_gpt_response(user_input):
    """接受用户输入，生成并执行查询，返回结果"""
    query = generate_database_query(user_input)
    if query:
        result = execute_query(query)
        return result
    else:
        return "Failed to generate or execute query."


while True:
    user_input = input("Please enter your database query: ")
    if user_input.lower() == 'exit':
        break
    response = sql_query_gpt_response(user_input)
    print(response)
