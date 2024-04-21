import os
import json
from antlr4 import CommonTokenStream, FileStream

from Python3Lexer import Python3Lexer
from Python3Parser import Python3Parser

from MyPython3ParserVisitor import MyPython3ParserVisitor


def parse_file(source_file):
    lexer = Python3Lexer(FileStream(fileName=source_file, encoding='utf8'))
    stream = CommonTokenStream(lexer)
    parser = Python3Parser(stream)
    visitor = MyPython3ParserVisitor()
    content = json.dumps(visitor.visitFile_input(parser.file_input()), indent=2)
    ast_file = os.path.splitext(source_file)[0] + '.json'
    with open(ast_file, 'wt', encoding='utf8') as jf:
        jf.write(content)


if __name__ == '__main__':
    for (dir, dirnames, files) in os.walk("../code_data"):
        for file in files:
            if file.endswith('.py'):
                print(f'parse file {file}')
                parse_file(os.path.join(dir, file))
