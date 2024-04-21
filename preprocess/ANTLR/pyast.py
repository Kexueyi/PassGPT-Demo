import os
import json
import ast2json


def parse_file(source_file):
    import ast
    with open(file=source_file, mode='rt', encoding='utf8') as f:
        tree = ast.parse(f.read())
        ast_file = os.path.splitext(source_file)[0] + '.json'
        print(f'{source_file} --> {ast_file}')
        with open(ast_file, 'wt', encoding='utf8') as jf:
            jf.write(json.dumps(ast2json.ast2json(tree)))


if __name__ == '__main__':
    for (dir, dirnames, files) in os.walk("../parse test cases"):
        for file in files:
            print(f'parse file {file}')
            try:
                parse_file(os.path.join(dir, file))
            except Exception as e:
                print('error:', e)
