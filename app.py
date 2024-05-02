import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the music dataset
music_data = pd.read_csv('music_dataset.csv')

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the text features
tfidf_matrix = vectorizer.fit_transform(music_data['features'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get recommendations based on a given music title
def get_recommendations(title, cosine_sim, music_data):
    # Get the index of the music title
    idx = music_data[music_data['title'] == title].index[0]

    # Get the pairwise similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the music titles based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top 5 recommendations
    top_recommendations = sim_scores[1:6]

    # Get the recommended music titles
    recommended_music = [music_data.iloc[i[0]]['title'] for i in top_recommendations]

    return recommended_music

# Streamlit app
def main():
    st.title("Music Recommender")

    # Get user input
    user_input = st.text_input("Enter a music title:")

    if st.button("Get Recommendations"):
        if user_input:
            recommendations = get_recommendations(user_input, cosine_sim, music_data)
            st.write("Recommended Music:")
            for recommendation in recommendations:
                st.write(recommendation)
        else:
            st.write("Please enter a music title.")

if __name__ == '__main__':
    main()
