import json


def load_course_info(file_path):
    """
    Loads the course information from a JSON file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        course_info = json.load(file)
    return course_info


def ner_summarize(json_data):
    """
    Summarizes the entities from the JSON data by listing text, labels, and page numbers.
    """
    summaries = []
    for document_title, content in json_data.items():
        document_summary = f"In the document '{document_title}', the following entities are mentioned:"
        entity_summaries = []

        for section, entities in content.items():  # Assuming 'content' is a dictionary containing lists of entities
            for entity in entities:  # Iterating through list of entities
                if isinstance(entity, dict):  # Checking if the entity is indeed a dictionary
                    text = entity.get('text', 'No Text Available')
                    label = entity.get('label', 'No Label Available')
                    pages = ", ".join(map(str, entity.get('page_num', [])))
                    entity_summary = f"{text} ({label}) on page {pages}"
                    entity_summaries.append(entity_summary)
                else:
                    continue  # If it's not a dictionary, skip to the next item

        document_summary += " " + "; ".join(entity_summaries) + "."
        summaries.append(document_summary)

    return " ".join(summaries)


def lda_summarize(json_data):
    """
    Summarizes the LDA topics from the JSON data by listing the top terms with their respective weights per topic.
    """
    summaries = []
    for document_title, topics in json_data.items():
        document_summary = f"The document '{document_title}' discusses the following topics:"
        topic_summaries = []

        for topic in topics:
            topic_id = list(topic.keys())[0]  # Getting the topic number (key of the dictionary)
            terms = topic[topic_id]

            # Prepare term summaries without using complex expressions inside f-strings
            term_details = []
            for term in terms.split('+'):
                weight, word = term.split('*')
                weight = weight.strip()
                word = word.strip().replace('"', '')  # Remove quotes around the word

                term_detail = f"{word} ({weight})"
                term_details.append(term_detail)

            terms_summary = ", ".join(term_details)
            topic_summary = f"Topic {topic_id}: {terms_summary}"
            topic_summaries.append(topic_summary)

        document_summary += " " + "; ".join(topic_summaries) + "."
        summaries.append(document_summary)

    return " ".join(summaries)


def dp_summarize(json_data):
    """
    Summarizes the dependency relationships from the JSON data for each sentence.
    """
    summaries = []
    for document_id, sentences in json_data.items():
        document_summary = f"Document ID '{document_id}' has the following sentences and their dependency structures:"
        sentence_summaries = []

        for sentence_data in sentences:
            sentence = sentence_data['sentence'].replace("\n", " ")  # Clean new lines for better readability
            tokens_summary = []

            for token in sentence_data['tokens']:
                text = token['text'].strip()
                if text == "\n" or text == "\n\n":  # Skip pure newline tokens for summary
                    continue
                dep = token['dep']
                head_text = token['head_text']
                head_pos = token['head_pos']

                token_summary = f"'{text}' ({dep}) depends on '{head_text}' ({head_pos})"
                tokens_summary.append(token_summary)

            sentence_summary = f"Sentence: {sentence} - Tokens: " + "; ".join(tokens_summary)
            sentence_summaries.append(sentence_summary)

        document_summary += " " + " ".join(sentence_summaries)
        summaries.append(document_summary)

    return " ".join(summaries)


file_path = "F:/NTU Assignment/EE6405/dp_results.json"
course_info = load_course_info(file_path)
course_info_summary = dp_summarize(course_info)

print(course_info_summary)
