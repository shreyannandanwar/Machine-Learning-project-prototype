import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.multioutput import MultiOutputRegressor
import plotly.express as px


@st.cache_data
def prepare_data(n_movies=1000):
    ratings = pd.read_csv('ml-latest-small/ratings.csv')
    movies = pd.read_csv('ml-latest-small/movies.csv')

    # Merge datasets
    df = pd.merge(ratings, movies, on='movieId')

    # Randomly sample movieIds
    sampled_movie_ids = df['movieId'].drop_duplicates().sample(n=n_movies, random_state=None)
    df = df[df['movieId'].isin(sampled_movie_ids)]

    # Pivot table for user-movie ratings
    matrix = df.pivot_table(index='userId', columns='title', values='rating').fillna(0)

    return matrix, df


def train_model(user_ratings, movie_matrix, sample_movies):

    # Prepare input: all users who rated all sample_movies
    df_train = movie_matrix[sample_movies].copy()
    df_train = df_train[df_train[sample_movies].sum(axis=1) > 0]  # avoid zero-only rows

    X_train = df_train[sample_movies]
    y_train = movie_matrix.drop(columns=sample_movies).loc[X_train.index]

    model = MultiOutputRegressor(DecisionTreeRegressor(max_depth=5))
    model.fit(X_train, y_train)

    # Prepare user vector
    user_vector = pd.DataFrame([user_ratings], columns=sample_movies)
    predicted_ratings = model.predict(user_vector)[0]

    unrated_movie_titles = movie_matrix.drop(columns=sample_movies).columns
    recommendations = pd.Series(predicted_ratings, index=unrated_movie_titles).sort_values(ascending=False)

    return recommendations, model, sample_movies



st.set_page_config(page_title="🎬 Movie Recommender")

st.title("🎬 Personalized Movie Recommender")

movie_matrix, df = prepare_data()
sample_movies = list(movie_matrix.columns[:10])

st.subheader("Step 1: Rate a few movies")
user_ratings = []
for movie in sample_movies:
    rating = st.slider(f"Rate: \n{movie}", 0.0, 5.0, 0.0, 0.5)
    user_ratings.append(rating)

if st.button("Get Recommendations"):
    recommended, model, feature_names = train_model(user_ratings, movie_matrix, sample_movies)

    st.subheader("Top Recommendations 🎯")
    st.write(recommended.head(5))

    fig = px.bar(recommended.head(10), x=recommended.head(10).values, y=recommended.head(10).index,
                 orientation='h', title="Top 10 Recommendations")
    st.plotly_chart(fig)

    # 🌳 Decision Tree Visualization - Hidden in Expander
    st.subheader("🧠 How the Decision Tree Makes Predictions")
    with st.expander("🔍 Show Decision Tree Visualization"):
        plt.figure(figsize=(15, 10))
        plot_tree(model.estimators_[0], feature_names=feature_names, filled=True, rounded=True)
        st.pyplot(plt.gcf())

st.markdown("---")
st.markdown("Made with ❤️ by [Shreyan Nandanwar](https://github.com/shreyannandanwar)")
