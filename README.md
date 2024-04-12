# PassGPT
A course-specific chat model powered by GPT. Make you learn efficiently and Pass! Let's meet PassGPT!

![Working Flow of PassGPT](figure/intro.png)

## Environment
The packages are listed in requirements.txt. Run the following command for setting up the environment:
```bash
conda create --name pgpt --file requirements.txt
conda activate pgpt
```

## Motivation
1. To mitigate the gap between general version of GPT and course-specific GPT
2. To provide a more efficient way to generate course-specific content
3. To provide a more interactive way to learn

## Functions
1. Generate course-specific content. e.g. course structure, concepts, and examples
2. Course information retrieval. e.g. course schedule, course materials, and course announcements
3. Study & review assistance. e.g. summarize, simulate quizzes

## Working Flow
### Data Preprocssing
#### Goal
We need course available information in a well-structed data form：
1. Material Type: slides, codes, announcements, quizzes
2. Teaching Week: week 1, week 2, ..., week n
3. Topic
4. Subtopic
5. Key Concepts
6. Formula
7. Code Example

#### Database
1. Course Material Database
2. Course Information Database
3. Course Quiz Database
4. Course Announcement Database
5. Code Material Database