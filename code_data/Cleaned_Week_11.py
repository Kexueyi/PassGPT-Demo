#!/usr/bin/env python
# coding: utf-8

# # Recommendation Chatbot with ChatGPT

# This is a simple version of the Recommendation Chatbot discussed in class.
# In this version, we manually input some samples for products and customer history. In a real system, these will be connected to existing databases. 
# This sample is an updated version of a tutorial found at https://github.com/norahsakal/chatgpt-product-recommendation-embeddings



#import the basic packages and functions that we need to use
from openai import OpenAI
import pandas as pd
from scipy.spatial.distance import cosine


# # 1. Setting up



#Input your API Key here, just like we did in Lecture 10
#If you do not have api_key, you can still follow the code and learn 
api_key =""
client = OpenAI(api_key = api_key)

#we need to define two functions that are used throughout this exercise

#one is the get_embedding function which helps us connect to OPENAI and get an embedding
def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

#the second to use the cosine similarity we already seen in lecture 10. 
#This time we define our own cosine_similarity directly from the cosine distance
#the cosine function imported gives us the cosine distance, the similarity is 1-cosine distance
def cosine_similarity(x, y):
    return 1 - cosine(x,y)


# # 2. Create product data



#define some products, this is to mimic our product database
product_data = [{
    "prod_id": 1,
    "prod": "moisturizer",
    "brand":"Aveeno",
    "description": "for dry skin"
},
{
    "prod_id": 2,
    "prod": "foundation",
    "brand":"Maybelline",
    "description": "medium coverage"
},
{
    "prod_id": 3,
    "prod": "moisturizer",
    "brand":"CeraVe",
    "description": "for dry skin"
},
{
    "prod_id": 4,
    "prod": "nail polish",
    "brand":"OPI",
    "description": "raspberry red"
},
{
    "prod_id": 5,
    "prod": "concealer",
    "brand":"chanel",
    "description": "medium coverage"
},
{
    "prod_id": 6,
    "prod": "moisturizer",
    "brand":"Ole Henkrisen",
    "description": "for oily skin"
},
{
    "prod_id": 7,
    "prod": "moisturizer",
    "brand":"CeraVe",
    "description": "for normal to dry skin"
},
{
    "prod_id": 8,
    "prod": "moisturizer",
    "brand":"First Aid Beauty",
    "description": "for dry skin"
},{
    "prod_id": 9,
    "prod": "makeup sponge",
    "brand":"Sephora",
    "description": "super-soft, exclusive, latex-free foam"
}]




#put the product into a pandas dataframe
product_data_df = pd.DataFrame(product_data)
#this is to show how it looks like in a dataframe format
product_data_df




#we add a column called "combined", this is essentially a concatenation of the values in each row into one.
#this will be used to generate our embeddings 
product_data_df['combined'] = product_data_df.apply(lambda row: f"{row['brand']}, {row['prod']}, {row['description']}", axis=1)
product_data_df




#this calls the openai api and use a particular engine to generate embedding
product_data_df['text_embedding'] = product_data_df.combined.apply(lambda x: get_embedding(x))
product_data_df


# # 3. Create customer profile data



#define some customer history, this is to mimic our customer history database
#note for simplicity, we only created history for one customer
customer_order_data = [
{
    "prod_id": 1,
    "prod": "moisturizer",
    "brand":"Aveeno",
    "description": "for dry skin"
},{
    "prod_id": 2,
    "prod": "foundation",
    "brand":"Maybelline",
    "description": "medium coverage"
},{
    "prod_id": 4,
    "prod": "nail polish",
    "brand":"OPI",
    "description": "raspberry red"
},{
    "prod_id": 5,
    "prod": "concealer",
    "brand":"chanel",
    "description": "medium coverage"
},{
    "prod_id": 9,
    "prod": "makeup sponge",
    "brand":"Sephora",
    "description": "super-soft, exclusive, latex-free foam"
}]




#similar to product, we convert the data into a dataframe
customer_order_df = pd.DataFrame(customer_order_data)
customer_order_df




#as before, we create a combined column to be used for embedding
customer_order_df['combined'] = customer_order_df.apply(lambda row: f"{row['brand']}, {row['prod']}, {row['description']}", axis=1)
customer_order_df




#as before, we create an embedding
customer_order_df['text_embedding'] = customer_order_df.combined.apply(lambda x: get_embedding(x))
customer_order_df


# # 4. Customer input



#create a sample customer input message
customer_input = "Hi! Can you recommend a good moisturizer for me?"




#turn it into an embedding
response = client.embeddings.create(
    input=customer_input,
    model="text-embedding-ada-002"
)
embeddings_customer_question = response.data[0].embedding


# # 5.  Find similar product from customer history and product list



#just like lecture 10, we use cosine similarity to find close matches in customer's purchase history
customer_order_df['search_purchase_history'] = customer_order_df.text_embedding.apply(lambda x: cosine_similarity(x, embeddings_customer_question))
customer_order_df = customer_order_df.sort_values('search_purchase_history', ascending=False)
customer_order_df




#we limit to the top 3 choices from purchase history
top_3_purchases_df = customer_order_df.head(3)
top_3_purchases_df




#now we find the similar products from our product list
product_data_df['search_products'] = product_data_df.text_embedding.apply(lambda x: cosine_similarity(x, embeddings_customer_question))
product_data_df = product_data_df.sort_values('search_products', ascending=False)
product_data_df




#we restrict to top 3, these top three are what we will recommend to the customer
top_3_products_df = product_data_df.head(3)
top_3_products_df


# # 6. Prompt Engineering for ChatGPT

# We create a large message object that does following:
# 1. Provide user input
# 2. Provide user purchase history (top 3)
# 3. provide top 3 product recommendation
# Throughout, we also ensure that we prompt ChatGPT to give appropriate response in the correct manner. 

# >Tip 💡
# >
# >Tinker with the instructions in the prompt until you find the desired voice of your chatbot.



#Create message object and setting the stage
message_objects = []
message_objects.append({"role":"system", "content":"You're a chatbot helping customers with beauty-related questions and helping them with product recommendations"})




# Append the customer message
message_objects.append({"role":"user", "content": customer_input})




# Create previously purchased input
prev_purchases = ". ".join([f"{row['combined']}" for index, row in top_3_purchases_df.iterrows()])
prev_purchases




# Append prev relevant purchase
message_objects.append({"role":"user", "content": f"Here're my latest product orders: {prev_purchases}"})
message_objects.append({"role":"user", "content": f"Please give me a detailed explanation of your recommendations"})
message_objects.append({"role":"user", "content": "Please be friendly and talk to me like a person, don't just give me a list of recommendations"})




# Create list of 3 products to recommend
products_list = []

for index, row in top_3_products_df.iterrows():
    brand_dict = {'role': "assistant", "content": f"{row['combined']}"}
    products_list.append(brand_dict)
products_list




# Append found products  
message_objects.append({"role": "assistant", "content": f"I found these 3 products I would recommend"})
message_objects.extend(products_list)
message_objects.append({"role": "assistant", "content":"Here's my summarized recommendation of products, and why it would suit you:"})
message_objects


# # 16. Call ChatGPT API



#Call ChatGPT and provide our prompt and get back the response
completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=message_objects
)
print(completion.choices[0].message.content)






