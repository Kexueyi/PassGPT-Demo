from openai import OpenAI
import pymysql
import os

class PassGPT:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API"))
        self.db_params = {
            'host': 'pass-gpt-db.c3oum0k4qlkb.ap-southeast-1.rds.amazonaws.com',
            'port': 3306,
            'user': 'root',
            'password': os.getenv("AWS_IRD_ROOT_KEY"),
            'database': 'passgpt',
            'charset': 'utf8mb4',
            'cursorclass': pymysql.cursors.DictCursor
        }

    def get_db_connection(self):
        return pymysql.connect(**self.db_params)

    def clean_sql_query(self, sql_query):
        sql_query = sql_query.replace("`", "").replace("sql\n", "").replace("\n", " ")
        return ' '.join(sql_query.split())

    def sql_query_gpt(self, user_input):
        connection = self.get_db_connection()
        database_description = """
        Database 'passgpt' consists of several tables: 'courses', 'files', and 'concepts'.
        - The 'courses' table has columns: course_id (primary key), course_code, course_name, course_description.
        - The 'files' table relates to 'courses' via course_id and includes: file_id (primary key), file_name, title, file_type, file_path, teaching_week, creation_date.
        - The 'concepts' table includes concepts that relate to files and can be hierarchical: concept_id (primary key), parent_id, file_id, concept_name, concept_page, concept_description.
        Relations:
        - 'files' to 'courses' via course_id.
        - 'concepts' to 'files' via file_id and 'concepts' to other 'concepts' via parent_id.
        """
        messages = [
            {"role": "system", "content": database_description},
            {"role": "system", "content": """You are an AI trained to generate SQL queries based on user's keywords given the database schema, then retrieve all relevant data with detailed information.
                                            If the user input is very vague, you may enrich the content, then querying major contents like files and main concepts.
                                            Please only generate clean SQL queries without explanations."""},
            {"role": "user", "content": user_input}
        ]
        try:
            chat_completion = self.client.chat.completions.create(
                model="gpt-4-turbo",  
                messages=messages
            )
            if chat_completion.choices and len(chat_completion.choices) > 0:
                sql_query_cmd = self.clean_sql_query(chat_completion.choices[0].message.content.strip())
                with connection.cursor() as cursor:
                    cursor.execute(sql_query_cmd)
                    sql_query_results = cursor.fetchall()
                connection.close()
                return sql_query_cmd, sql_query_results
            else:
                connection.close()
                return "No response was generated by GPT."
        except Exception as e:
            connection.close()
            return "No valid SQL query could be generated. Please specify your question more precisely."

    def course_query_gpt(self, query, course_info):
        messages = [
            {"role": "system", "content": f"""{course_info}This is course_info provided to you. 
             Your task is to provide a comprehensive response based on the course_info provided, describe the informations in a detailed, human-readable form for education need. 
             You need to provide a detailed explanation of terms with examples, including the main concepts, related sub-concepts and also pages. You may need to analysis difficulty, provide tips and estimate learning time to consume related course_info. 
             Only if you are asked to provide quiz/question/test, then generate 5 questios in each 2 types related to course_info: 
             1.code questions: single choice questions testing understanding of the code representing related concepts, provided with code segment. 
             2. context questions: multiple choice testing understanding of all concepts. 
             Finally, you should give the answer."""},
            {"role": "user", "content": query}
        ]
        try:
            chat_completion = self.client.chat.completions.create(
                model="gpt-4-turbo",  
                messages=messages
            )
            if chat_completion.choices and len(chat_completion.choices) > 0:
                last_message = chat_completion.choices[0].message.content.strip()
                return last_message
            else:
                return "No response was returned by GPT."
        except Exception as e:
            return f"Error querying GPT with course info: {e}"
