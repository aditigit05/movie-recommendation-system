"""
========================================================
            MOVIE RECOMMENDATION SYSTEM
========================================================

Tech Stack:
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity

Description:
A content-based movie recommendation system that
suggests similar movies using machine learning
similarity techniques.

========================================================
"""

# ------------------------------------------------------
# IMPORT LIBRARIES
# ------------------------------------------------------

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------------------
# MOVIE DATASET
# ------------------------------------------------------

movies_data = {

    "title": [

        "Inception",
        "Interstellar",
        "The Dark Knight",
        "Titanic",
        "Avengers Endgame",
        "Doctor Strange",
        "Avatar",
        "Iron Man",
        "Batman Begins",
        "The Prestige"
    ],

    "genre": [

        "Sci-Fi Thriller",
        "Sci-Fi Space Adventure",
        "Action Crime Superhero",
        "Romance Drama",
        "Action Superhero Adventure",
        "Fantasy Magic Superhero",
        "Sci-Fi Adventure Fantasy",
        "Technology Action Superhero",
        "Action Crime Batman",
        "Mystery Thriller Drama"
    ]
}


# ------------------------------------------------------
# CREATE DATAFRAME
# ------------------------------------------------------

movies_df = pd.DataFrame(movies_data)


# ------------------------------------------------------
# FEATURE EXTRACTION
# ------------------------------------------------------

vectorizer = TfidfVectorizer()

feature_vectors = vectorizer.fit_transform(
    movies_df["genre"]
)


# ------------------------------------------------------
# CALCULATE SIMILARITY
# ------------------------------------------------------

similarity_matrix = cosine_similarity(
    feature_vectors
)


# ------------------------------------------------------
# MOVIE RECOMMENDATION FUNCTION
# ------------------------------------------------------

def recommend_movies(movie_name):

    movie_name = movie_name.lower()

    movie_titles = movies_df["title"].str.lower().tolist()

    # Check movie existence
    if movie_name not in movie_titles:

        print("\nMovie not found in database.")
        return

    # Get movie index
    movie_index = movie_titles.index(movie_name)

    # Similarity scores
    similarity_scores = list(
        enumerate(similarity_matrix[movie_index])
    )

    # Sort based on similarity
    sorted_movies = sorted(

        similarity_scores,

        key=lambda x: x[1],

        reverse=True
    )

    # Display recommendations
    print("\n===================================")
    print("     Recommended Movies")
    print("===================================\n")

    for movie in sorted_movies[1:6]:

        index = movie[0]

        recommended_movie = movies_df.iloc[index]["title"]

        similarity_score = round(
            movie[1] * 100,
            2
        )

        print(
            f"{recommended_movie}"
            f"  |  Similarity Score: "
            f"{similarity_score}%"
        )


# ------------------------------------------------------
# USER INPUT
# ------------------------------------------------------

print("\n========== Movie Recommendation System ==========\n")

movie_input = input("Enter Movie Name: ")

recommend_movies(movie_input)