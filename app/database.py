import pymysql
import os
from pymysql.cursors import DictCursor

def get_db_connection():
    return pymysql.connect(host='pass-gpt-db.c3oum0k4qlkb.ap-southeast-1.rds.amazonaws.com',
                           port=3306,
                           user='root', 
                           password= os.getenv("AWS_IRD_ROOT_KEY"), 
                           database='passgpt',
                           charset='utf8mb4',
                           cursorclass=DictCursor)

