# PassGPT
A course-specific chat model powered by GPT. Make you learn efficiently and Pass! Let's meet PassGPT!

![Working Flow of PassGPT](figure/intro.png)

This repo is mainly for course project from EE6405: Natural Language Processing. The goal of this project is to build a course-specific chat model powered by GPT. The model utlize the dataset build upon the course materials from EE6405.

## Main Functions
1. Course information retrieval. e.g. course materials, concepts related informations
2. Generate course-specific content. e.g. examples, sample practices, quiz
3. Study & review assistance. e.g. summarize, tips 


## Repository Overview
- `data/`: note that we are not able to release the original course matrials due to the policy.
  - `slides`: slides from EE6405
  - `code/`: code material from EE6405
  - `json/`: extracted structed course `.json` file from GPT-4 also traditional NLP methods.
- `database/`:
  - `json_to_sql`: convert `.json` to insert samples into SQL database.
  - `pass_gpt`: You can construct same database by running this script.
- `example_results/`: Example results of PassGPT, where some of them are good and some are not so satisfying due to not clear prompt from user.
- `models/`: Models for PassGPT.
  - `demo_json`: A demo just using json file as course information.
  - `demo_sql`: A demo using SQL database as course information.
  - `pass_gpt`: The main model for PassGPT.
- `web/`: Web interface for PassGPT. (under construction)
- `traditional_nlp/`: Traditional NLP methods for course information retrieval.
- `main.py`: Main script for running PassGPT.
- `requirements.txt`: List of packages required to run the code.


## Setup
### Environment
The packages are listed in requirements.txt. Run the following command for setting up the environment:
```bash
conda create --name pgpt --file requirements.txt
conda activate pgpt
```

### Database
You can construct your own `MySQL` database by running the following command:
```bash
mysql -u root -p < database/pass_gpt.sql
```

### OpenAI API
You need to have an OpenAI API key to run the code. You can get one from [OpenAI](https://beta.openai.com/signup/). Once you have the key, you can set it as an environment variable:
```bash
vim ~/.bashrc
export OPENAI_API_KEY="your-api-key" # add this line in your bashrc
# :wq
source ~/.bashrc
```

### Run
You can run the code in your terminal by:
```bash
python main.py
```
Then you can input your questions or commands to PassGPT. Enjoy!
