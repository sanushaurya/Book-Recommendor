import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load dataset
books = pd.read_csv("data/books.csv")
books['description'] = books['description'].fillna('')

# Vectorize descriptions
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['description'])

def get_recommendations_by_description(input_desc):
    input_tfidf = tfidf.transform([input_desc])  # Vectorize the input
    cosine_similarities = linear_kernel(input_tfidf, tfidf_matrix).flatten()
    
    # Get top 5 most similar books
    top_indices = cosine_similarities.argsort()[-5:][::-1]
    
    recommendations = []
    for idx in top_indices:
        title = books.iloc[idx]['title']
        desc = books.iloc[idx]['description']
        recommendations.append(f"📖 {title}\n{desc[:300]}...\n")
    
    return "\n".join(recommendations)
