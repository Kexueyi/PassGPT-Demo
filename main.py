from colorama import Fore, Style, init
from models.pass_gpt import PassGPT  

init(autoreset=True)  

def main():
    print(Style.BRIGHT + Fore.BLUE + """
    ________                         _________________ ________
    ___  __ \______ ___________________  ____/___  __ \___  __/
    __  /_/ /_  __ `/__  ___/__  ___/_  / __  __  /_/ /__  /   
    _  ____/ / /_/ / _(__  ) _(__  ) / /_/ /  _  ____/ _  /    
    /_/      \__,_/  /____/  /____/  \____/   /_/      /_/     
                                                                                                                
    """ + Style.RESET_ALL)

    processor = PassGPT()  # Our PassGPT model class
   
    while True:
        query = input("Enter your message to PassGPT:")
        if query.lower() == 'exit':
            break
        try:
            query_cmd, query_results = processor.sql_query_gpt(query)
            if query_results:
                response = processor.course_query_gpt(query, str(query_results))
            else:
                response = "No valid SQL results to process."
            print(response)
        except Exception as e:
            print(Fore.RED + f"Error: {e}")

if __name__ == '__main__':
    main()
