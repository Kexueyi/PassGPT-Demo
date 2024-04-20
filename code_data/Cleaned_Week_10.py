#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis and Movie Plot Generation using ChatGPT

# The goal of this script is to learn how to leverage OpenAI's API and prompt ChatGPT to perform meaningful tasks. 
# We will perform the following tasks:
# 1. Extract movie review data
# 2. Prompt ChatGPT to label the reviews (positive or negative sentiment)
# 3. Check the accuracy of ChatGPT's labels
# 4. Compute what typical reviews look like by using embeddings
# 5. Ask ChatGPT to extract useful sentences from typical positive reviews
# 6. Based on the extracted sentences, ask Chatgpt to summarize what makes these reviews positive
# 7. Based on the summary, ask ChatGPT to create a movie plot that will likely result in positive

# # Prepare movie review data



from torchtext.datasets import IMDB
import os

#function to load IMBD reviews, we default to 10 samples of positive and negative
def load_imdb_data(n_samples=10):
    positive_reviews, negative_reviews = [], []
    
    train_iter = IMDB(split='train')
    
    for label, text in train_iter:
        if label == 2 and len(positive_reviews) < n_samples:  # Assuming 2 is positive
            positive_reviews.append(text)
        elif label == 1 and len(negative_reviews) < n_samples:  # Assuming 1 is negative
            negative_reviews.append(text)
        if len(positive_reviews) >= n_samples and len(negative_reviews) >= n_samples:
            break

    return positive_reviews, negative_reviews

positive_reviews, negative_reviews = load_imdb_data()

# Verify the data
print(f"Loaded {len(positive_reviews)} positive reviews.")
print(f"Loaded {len(negative_reviews)} negative reviews.")


# # Use ChatGPT to predict sentiment of the reviews



from openai import OpenAI
import os
import random

# Set your OpenAI API key here
client = OpenAI(api_key="INSERT API KEY")

def predict_sentiment(reviews):
    sentiments = []
    for review in reviews:
        try:
            # Use the new client interface for chat completions
            chat_completion = client.chat.completions.create(
                model="gpt-3.5-turbo",  # Adjust the model as needed
                messages=[
                    {"role": "system", "content": "You are a helpful assistant trained to classify sentiment as positive or negative. Provide response with one word either 'postive' or 'negative'"},
                    {"role": "user", "content": review}
                ],
            )
            # Extract the sentiment from the completion's response
            sentiment = chat_completion.choices[0].message.content.strip().lower()
            #testing code can be removed
            print(sentiment)
            # Ensure sentiment is explicitly classified as 'positive' or 'negative'
            if sentiment not in ['positive', 'negative']:
                sentiment = random.choice(['positive', 'negative'])
            sentiments.append(sentiment)
        except Exception as e:
            print(f"Error processing review: {e}")
            # In case of an error, make a random decision between 'positive' and 'negative'
            sentiments.append(random.choice(['positive', 'negative']))
    return sentiments

# Example usage with your lists of reviews
positive_sentiments = predict_sentiment(positive_reviews)
negative_sentiments = predict_sentiment(negative_reviews)


# # Check OpenAI performance



#since we have the true labels for our reviews, we can actually check how well ChatGPT performs!

from sklearn.metrics import precision_score, recall_score, f1_score

# True labels: 1 for positive, 0 for negative
true_labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)
# Predicted labels: 1 for positive, 0 for negative
predicted_labels = [1 if sentiment == 'positive' else 0 for sentiment in positive_sentiments + negative_sentiments]

# Calculate metrics
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")


# # Find typical review



# Function to compute embeddings for a list of reviews
def compute_embeddings(reviews):
    embeddings = []
    for review in reviews:
        try:
            response = client.embeddings.create(
              input=review,
              model="text-embedding-ada-002"
            )
            embeddings.append(response.data[0].embedding)
        except Exception as e:
            print(f"Error computing embedding for review: {e}")
            # In case of an error, append a zero vector
            embeddings.append([0] * 2048) # Assuming embedding size is 2048
    return embeddings

# Compute embeddings for all reviews
positive_embeddings = compute_embeddings(positive_reviews)
negative_embeddings = compute_embeddings(negative_reviews)




#now we try to a "typical good review" and "typical bad review"
#here we use the highest average cosine similarity to define "typical"
#in the sense that the "typical" one is on average most similar to the others

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

def compute_cosine_similarities(embeddings):
    # Compute the cosine similarity matrix
    cos_sim_matrix = cosine_similarity(embeddings)
    return cos_sim_matrix

def find_typical_review_with_dataframe(reviews, embeddings):
    """
    Computes pairwise cosine similarity, finds the mean similarity for each review,
    stores results in a DataFrame, and identifies the 'typical' review.
    
    Parameters:
    - reviews: List of review texts.
    - embeddings: Corresponding embeddings for each review.
    
    Returns:
    - DataFrame containing reviews and their average cosine similarity.
    - The 'typical' review based on the highest average cosine similarity.
    """
    cos_sim_matrix = compute_cosine_similarities(embeddings)
    
    # Calculate the mean cosine similarity for each review, excluding self-comparison
    mean_cos_sim = np.mean(cos_sim_matrix - np.eye(cos_sim_matrix.shape[0]), axis=1)
    
    # Create a DataFrame for reviews and their average cosine similarities
    df = pd.DataFrame({'Review': reviews, 'AvgCosSim': mean_cos_sim})
    
    # Sort the DataFrame to find the review with the highest average cosine similarity
    df_sorted = df.sort_values(by='AvgCosSim', ascending=False)
    
    # The 'typical' review is the one with the highest average cosine similarity
    typical_review = df_sorted.iloc[0]['Review']
    
    return df_sorted, typical_review

# Example usage:
# Assume `positive_reviews`, `negative_reviews`, `positive_embeddings`, `negative_embeddings` are defined

# Compute for positive reviews
positive_df, typical_positive_review = find_typical_review_with_dataframe(positive_reviews, positive_embeddings)

# Compute for negative reviews
negative_df, typical_negative_review = find_typical_review_with_dataframe(negative_reviews, negative_embeddings)




print("Typical Positive Review:", typical_positive_review)




print("Typical Negative Review:", typical_negative_review)


# # Summarize Commonalities



top_5_positive_reviews = positive_df.head(5)['Review'].tolist()




def extract_positive_sentences(reviews):
    extracted_sentences = []
    for review in reviews:
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Extract 3 sentences from the following review that best indicate a positive sentiment:'"},
                    {"role": "user", "content": review}
                ],
            )
            sentences = response.choices[0].message.content.strip()
            extracted_sentences.extend(sentences.split('\n')[:3])  # Ensure only up to 3 sentences are added
        except Exception as e:
            print(f"Error extracting sentences: {e}")
    return extracted_sentences

positive_sentences = extract_positive_sentences(top_5_positive_reviews)




positive_sentences




def summarize_commonalities(sentences):
    aggregated_sentences = " ".join(sentences)
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Adjust model as necessary
            messages=[
                    {"role": "system", "content": "Summarize what the following sentences from movie reviews have in common, indicating good qualities of a movie:"},
                    {"role": "user", "content": aggregated_sentences}
                ],   
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error summarizing commonalities: {e}")
        return ""

commonalities_summary = summarize_commonalities(positive_sentences)




print("Commonalities Summary:", commonalities_summary)




def suggest_movie_plot(summary):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Adjust model as necessary
            messages=[
                    {"role": "system", "content": "Based on the following commonalities in good movie reviews, suggest a movie plot that is likely to result in a good review."},
                    {"role": "user", "content": summary}
                ], 
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error suggesting movie plot: {e}")
        return ""

movie_plot_suggestion = suggest_movie_plot(commonalities_summary)




print("Movie Plot Suggestion:", movie_plot_suggestion)






