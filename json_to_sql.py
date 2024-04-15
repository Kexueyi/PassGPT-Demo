import json

with open('data/outline/info.json', 'r') as file:
    data = json.load(file)

# 准备存储SQL命令的列表
sql_commands = []

def extract_pages(pages):
    return ', '.join(str(page) for page in pages)

# 处理课程和相关数据
for course_title, details in data.items():
    course_code = course_title.split('_')[0]
    title = details['Title']
    material_type = details['Material Type']
    teaching_week = details['Teaching Week'].replace('week ', '')
    file_name = course_title

    # 构建课程和文件的SQL插入命令
    sql_commands.append(f"INSERT INTO courses (course_code, course_name) VALUES ('{course_code}', 'Natural Language Processing');")
    sql_commands.append(f"INSERT INTO files (course_id, file_name, file_type, teaching_week) VALUES ((SELECT course_id FROM courses WHERE course_code='{course_code}'), '{file_name}', '{material_type}', {teaching_week});")
    
    # 处理概念
    for concept in details.get('Concepts', []):
        concept_name = concept['name']
        concept_pages = extract_pages(concept['Page'])

        # 插入概念
        sql_commands.append(f"INSERT INTO concepts (file_id, concept_name, concept_page) VALUES ((SELECT file_id FROM files WHERE file_name='{file_name}'), '{concept_name}', '{concept_pages}');")

        # 处理子概念和例子
        for subconcept in concept.get('Subconcepts', []):
            subconcept_name = subconcept['name']
            subconcept_pages = extract_pages(subconcept.get('Page', []))

            # 插入子概念
            sql_commands.append(f"INSERT INTO concepts (parent_id, file_id, concept_name, concept_page) VALUES ((SELECT concept_id FROM concepts WHERE concept_name='{concept_name}'), (SELECT file_id FROM files WHERE file_name='{file_name}'), '{subconcept_name}', '{subconcept_pages}');")
            
            # 处理例子
            for example in subconcept.get('Formula', []) + subconcept.get('Code', []):
                example_description = example['description']
                example_pages = extract_pages(example.get('Page', []))
                sql_commands.append(f"INSERT INTO examples (concept_id, example_name, example_page, example_description) VALUES ((SELECT concept_id FROM concepts WHERE concept_name='{subconcept_name}'), '{example_description}', '{example_pages}', '{example_description}');")

# 打印生成的SQL命令
for command in sql_commands:
    print(command)
