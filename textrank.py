import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import io

def calculate_sentence_scores(sentences):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sentences)
    X = normalize(X)
    similarity_matrix = cosine_similarity(X)
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    return scores

def text_rank_summary(text):
    sentences = text.split("\n")
    sentence_scores = calculate_sentence_scores(sentences)
    ranked_sentences = sorted(((sentence_scores[i], sentence) for i, sentence in enumerate(sentences)), reverse=True)
    summary = " ".join([sentence for score, sentence in ranked_sentences[:3]])  # Get the top 3 sentences as summary
    return summary

