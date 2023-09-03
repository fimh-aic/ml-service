# Import Dependencies
import pandas as pd
import numpy as np
import os
import requests, re, string
import difflib
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory as SF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Read dataset to get the result
df_resep = pd.read_csv('app/data/data_resep.csv')

# Read dataset for processing
df_cleaned = pd.read_csv('app/data/data_resep_cleaned.csv')
df_cleaned['tags'] = df_cleaned['menu'] + ' ' + df_cleaned['bahan'] + ' ' + df_cleaned['langkah']

# Make object Tfidf and Cosine Similarity for searching the desired resep
vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1,2))
features_vectors = vectorizer.fit_transform(df_cleaned['tags'])
similarity = cosine_similarity(features_vectors)

# Function to clean the user input
def case_folding(data):
    data = data.lower()
    data = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",data).split())
    data = re.sub(r"\d+", "", data)
    data = data.translate(str.maketrans("","",string.punctuation))
    data = re.sub(r"\n","",data)
    data = re.sub(r"\t","",data)
    keyword_list = [keyword.strip() for keyword in data.split()]
    return keyword_list


def stopword_cleaner(data):
    sw_indonesia = stopwords.words("indonesian")
    sw_indonesia.remove('tahu')
    data  = [word for word in data if word not in sw_indonesia]
    data = ' '.join(data)
    return data

# Function to process user input and get the result
def recommend(keywords, limit):

    # Initialize an empty dictionary to store the combined similarity scores for each recipe index
    combined_scores = {}
    
    # Search each keyword separately and calculate combined similarity scores
    for keyword in keywords:
        # Search the keyword in df_cleaned["menu"]
        menu_similarity_scores = [difflib.SequenceMatcher(None, keyword, menu).ratio() for menu in df_cleaned['menu'].tolist()]
        menu_closest_match_index = menu_similarity_scores.index(max(menu_similarity_scores))
        
        # Keyword matching for "bahan"
        keyword_matched_indices = [idx for idx, bahan in enumerate(df_cleaned['bahan']) if keyword in bahan]
        
        # Calculate similarity scores for "bahan" using TfidfVectorizer
        bahan_vectors = vectorizer.transform([keyword] + df_cleaned['bahan'].tolist())
        bahan_similarity_scores = cosine_similarity(bahan_vectors)[0, 1:]  # Similarity scores excluding the input keyword
        
        # Combine keyword matching and cosine similarity scores
        for idx in keyword_matched_indices:
            combined_score = 0.7 * menu_similarity_scores[idx] + 0.3 * bahan_similarity_scores[idx]
            if idx in combined_scores:
                combined_scores[idx] += combined_score
            else:
                combined_scores[idx] = combined_score
    
    # Sort the combined similarity scores in descending order
    sorted_scores = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Get the recommended recipes based on the combined scores
    result_list = []

    for idx, score in sorted_scores[:limit]:
        temp_data = df_resep.loc[idx].copy().to_dict()
        temp_data["bahan"] = ast.literal_eval(temp_data["bahan"])
        temp_data["langkah"] = ast.literal_eval(temp_data["langkah"])
        result_list.append(temp_data)
    
    return result_list