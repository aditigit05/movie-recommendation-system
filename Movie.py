import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------
# LOAD CSV DATASET
# -----------------------------------

df = pd.read_csv("movies.csv")


# -----------------------------------
# COMBINE FEATURES
# -----------------------------------

df["features"] = df["genre"] + " " + df["director"]


# -----------------------------------
# TF-IDF VECTORIZATION
# -----------------------------------

vectorizer = TfidfVectorizer()

feature_vectors = vectorizer.fit_transform(
    df["features"]
)


# -----------------------------------
# CALCULATE SIMILARITY
# -----------------------------------

similarity = cosine_similarity(
    feature_vectors
)


# -----------------------------------
# RECOMMENDATION FUNCTION
# -----------------------------------

def recommend_movies(movie_name):

    movie_name = movie_name.lower()

    movie_titles = df["title"].str.lower().tolist()

    if movie_name not in movie_titles:

        print("\nMovie not found.")
        return

    movie_index = movie_titles.index(movie_name)

    similarity_scores = list(
        enumerate(similarity[movie_index])
    )

    sorted_movies = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    print("\n========== Recommended Movies ==========\n")

    for movie in sorted_movies[1:6]:

        index = movie[0]

        title = df.iloc[index]["title"]

        rating = df.iloc[index]["rating"]

        director = df.iloc[index]["director"]

        similarity_score = round(
            movie[1] * 100,
            2
        )

        print(f"Movie: {title}")

        print(f"Director: {director}")

        print(f"Rating: {rating}")

        print(
            f"Similarity Score: "
            f"{similarity_score}%"
        )

        print("-----------------------------------")


# -----------------------------------
# USER INPUT
# -----------------------------------

movie_input = input(
    "Enter Movie Name: "
)

recommend_movies(movie_input)