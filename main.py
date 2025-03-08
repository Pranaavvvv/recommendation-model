from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('dataset2.csv')

# Extract base model name (e.g., "Samsung Galaxy M06 5G" from "Samsung Galaxy M06 5G (Sage Green, 6GB RAM, 128 GB Storage)")
df['base_model'] = df['name'].apply(lambda x: x.split('(')[0].strip())

# Combine text features into a single column
df['combined_text'] = df['name'] + ' ' + df['brand'] + ' ' + df['description'] + ' ' + df['search_query']

# Normalize numerical features (price and rating)
scaler = MinMaxScaler()
df[['price_normalized', 'rating_normalized']] = scaler.fit_transform(df[['price', 'rating']])

# TF-IDF for text features
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_text'])

# Combine TF-IDF and numerical features
numerical_features = df[['price_normalized', 'rating_normalized']].values
feature_matrix = hstack([tfidf_matrix, numerical_features])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(feature_matrix, feature_matrix)

# Function to get recommendations based on a product name
def get_recommendations(product_name, cosine_sim=cosine_sim, df=df, top_n=5):
    """
    Get top N recommendations for a product, ensuring recommendations are for different products (not just variants).
    """
    # Find the index of the selected product
    product_index = df[df['name'] == product_name].index[0]
    
    # Get the base model of the selected product
    base_model = df.loc[product_index, 'base_model']
    
    # Get similarity scores for the selected product
    sim_scores = list(enumerate(cosine_sim[product_index]))
    
    # Sort by similarity score (descending order)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Filter out products with the same base model and ensure unique products
    unique_recommendations = []
    seen_models = set()  # Track seen base models to avoid duplicates
    seen_models.add(base_model)  # Add the selected product's base model to avoid recommending it
    
    for i, score in sim_scores:
        current_base_model = df.loc[i, 'base_model']
        if current_base_model not in seen_models:  # Ensure it's a different product
            unique_recommendations.append(i)
            seen_models.add(current_base_model)  # Add to seen models to avoid duplicates
            if len(unique_recommendations) >= top_n:  # Stop after top N recommendations
                break
    
    # Get the names of the recommended products
    recommended_products = df.iloc[unique_recommendations][['name', 'price', 'images']].to_dict('records')
    return recommended_products

# API endpoint to get recommendations
@app.route('/recommend', methods=['GET'])
def recommend():
    product_name = request.args.get('product_name')
    if not product_name:
        return jsonify({"error": "Product name is required"}), 400
    
    recommendations = get_recommendations(product_name)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
