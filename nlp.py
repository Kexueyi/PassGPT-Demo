import spacy
import json
import os
import re
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBoxHorizontal
from pdfminer.high_level import extract_text
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models


class NERProcessor:
    def __init__(self, model='en_core_web_sm'):
        self.nlp = spacy.load(model)

    @staticmethod
    def extract_text_by_page(pdf_path):
        for page_layout in extract_pages(pdf_path):
            page_text = ""
            for element in page_layout:
                if isinstance(element, LTTextBoxHorizontal):
                    page_text += element.get_text()
            yield page_text.strip()

    def process_pdf(self, pdf_path):
        entities_by_concept = {}
        page_num = 0
        for page_text in self.extract_text_by_page(pdf_path):
            page_num += 1
            lines = page_text.split('\n')
            if lines:
                title = lines[0].strip()
                doc = self.nlp(page_text)
                for ent in doc.ents:
                    # Create a unique key by combining the entity text and label
                    entity_key = f'{ent.text}|{ent.label_}'
                    entity_info = {
                        "text": ent.text,
                        "label": ent.label_,
                        "page_num": [page_num]
                    }
                    # Initialize the title in entities_by_concept if not already present
                    if title not in entities_by_concept:
                        entities_by_concept[title] = {}
                    if entity_key not in entities_by_concept[title]:
                        entities_by_concept[title][entity_key] = entity_info
                    else:
                        # If the entity is already present, append the page number
                        entities_by_concept[title][entity_key]["page_num"].append(page_num)

        # Convert the keys back to lists of dictionaries
        formatted_entities_by_concept = {title: list(entities.values()) for title, entities in
                                         entities_by_concept.items()}
        return formatted_entities_by_concept

    def process_folder(self, pdf_folder):
        all_entities = {}
        for filename in os.listdir(pdf_folder):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(pdf_folder, filename)
                entities = self.process_pdf(pdf_path)
                all_entities[filename] = entities
        return all_entities

    @staticmethod
    def save_results(results, output_path):
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(results, file, indent=4, ensure_ascii=False)


class LDAprocesser:
    @staticmethod
    def preprocess(text):
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        words = word_tokenize(text.lower())
        processed_words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
        return processed_words

    @staticmethod
    def extract_topics(texts, num_topics=88, num_words=15):
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
        topics = lda_model.print_topics(num_words=num_words)
        return [{int(topic[0]): topic[1]} for topic in topics]

    def save_results(self, pdf_folder, output_path):
        texts = []
        pdf_titles = []
        for filename in os.listdir(pdf_folder):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(pdf_folder, filename)
                text = extract_text(pdf_path)
                processed_text = self.preprocess(text)
                texts.append(processed_text)
                pdf_titles.append(filename[:-4])

        topics = self.extract_topics(texts, num_topics=88, num_words=15)
        pdf_topics = {title: [] for title in pdf_titles}

        # Ensure each document receives topics
        topics_per_document = 8
        for idx, title in enumerate(pdf_titles):
            topic_indices = range(idx * topics_per_document, (idx + 1) * topics_per_document)
            assigned_topics = [topics[i % len(topics)] for i in topic_indices]  # Use modulo to avoid index error
            pdf_topics[title] = assigned_topics

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(pdf_topics, f, indent=4, ensure_ascii=False)


class DPprocesser:
    def __init__(self, model='en_core_web_sm'):
        self.nlp = spacy.load(model)

    @staticmethod
    def extract_text_from_pdf(pdf_path):
        return extract_text(pdf_path)

    def dependency_parse(self, text):
        doc = self.nlp(text)
        parsed_data = []
        for sent in doc.sents:
            sent_data = {"sentence": sent.text, "tokens": []}
            for token in sent:
                token_data = {
                    "text": token.text,
                    "dep": token.dep_,
                    "head_text": token.head.text,
                    "head_pos": token.head.pos_
                }
                sent_data["tokens"].append(token_data)
            parsed_data.append(sent_data)
        return parsed_data

    @staticmethod
    def get_week_from_filename(filename):
        match = re.search(r'W\d+', filename)
        return match.group(0) if match else 'Unknown'

    @staticmethod
    def save_data_to_json(data, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def save_results(self, folder_path, output_path):
        all_data = {}
        for filename in os.listdir(folder_path):
            if filename.endswith(".pdf"):
                week = self.get_week_from_filename(filename)
                pdf_path = os.path.join(folder_path, filename)
                text = self.extract_text_from_pdf(pdf_path)
                if text is not None:
                    parsed_data = self.dependency_parse(text)
                    if week not in all_data:
                        all_data[week] = []
                    all_data[week].extend(parsed_data)
        self.save_data_to_json(all_data, output_path)
