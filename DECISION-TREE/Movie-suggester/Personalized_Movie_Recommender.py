import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.multioutput import MultiOutputRegressor
import plotly.express as px


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



#
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.tree import DecisionTreeRegressor
# import plotly.express as px
#
# # ------------------------------
# # Caching large file loads
# # ------------------------------
# ratings = pd.read_csv("ml-latest/ratings.csv", dtype={
#     'userId': 'int32',
#     'movieId': 'int32',
#     'rating': 'float32',
#     'timestamp': 'int64'
# })
# @st.cache_data
# def load_data():
#     # Load movies
#     movies = pd.read_csv("ml-latest/movies.csv")
#
#     # Load large ratings.csv in chunks and filter
#     chunk_size = 500_000
#     chunks = []
#     for chunk in pd.read_csv("ml-latest/ratings.csv", chunksize=chunk_size, dtype={
#         'userId': 'int32',
#         'movieId': 'int32',
#         'rating': 'float32',
#         'timestamp': 'int64'
#     }):
#         filtered = chunk[chunk['userId'] <= 1000]  # keep first 1000 users
#         chunks.append(filtered)
#         if len(pd.concat(chunks)) > 1_000_000:
#             break  # stop once we have enough
#
#     ratings = pd.concat(chunks)
#     return ratings, movies
#
# # ------------------------------
# # Preprocess into pivot matrix
# # ------------------------------
# @st.cache_data
# def prepare_matrix(ratings):
#     # Pivot: userId x movieId
#     movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
#     return movie_matrix
#
# # ------------------------------
# # Train the decision tree model
# # ------------------------------
# def train_model(user_ratings, movie_matrix, sample_movies):
#     # Create one input row from user's ratings
#     user_input = pd.Series(0, index=movie_matrix.columns)
#     for movie_id, rating in user_ratings.items():
#         user_input[movie_id] = rating
#
#     # Add user to movie matrix
#     movie_matrix = movie_matrix.copy()
#     movie_matrix.loc[-1] = user_input
#     movie_matrix = movie_matrix.sort_index()
#
#     # Train on all rows except last one (user)
#     X = movie_matrix.drop(columns=sample_movies)
#     y = movie_matrix[sample_movies]
#
#     X_train = X.iloc[:-1]
#     y_train = y.iloc[:-1]
#     X_user = X.iloc[[-1]]
#
#     model = DecisionTreeRegressor(max_depth=5)
#     model.fit(X_train, y_train)
#
#     predictions = model.predict(X_user)[0]
#     return dict(zip(sample_movies, predictions))
#
# # ------------------------------
# # Main App
# # ------------------------------
# def main():
#     st.title("🎬 Personalized Movie Recommender")
#     st.write("Rate a few movies and get personalized recommendations using Decision Tree!")
#
#     # Load data
#     with st.spinner("Loading data..."):
#         ratings, movies = load_data()
#         movie_matrix = prepare_matrix(ratings)
#
#     # Select a few random movies for user to rate
#     popular_movies = ratings['movieId'].value_counts().head(1000).index.tolist()
#     sample_movies = np.random.choice(popular_movies, size=10, replace=False)
#     sample_df = movies[movies['movieId'].isin(sample_movies)]
#
#     st.subheader("🎯 Rate the following movies")
#     user_ratings = {}
#     for _, row in sample_df.iterrows():
#         rating = st.slider(f"{row['title']}", 1, 5, step=1)
#         user_ratings[row['movieId']] = rating
#
#     if st.button("🚀 Recommend Movies"):
#         with st.spinner("Training model..."):
#             predictions = train_model(user_ratings, movie_matrix, sample_movies)
#
#         # Show results
#         result_df = pd.DataFrame({
#             'movieId': list(predictions.keys()),
#             'Predicted Rating': list(predictions.values())
#         }).merge(movies, on='movieId')
#
#         result_df = result_df.sort_values(by='Predicted Rating', ascending=False)
#
#         st.subheader("🎉 Recommended for You")
#         st.dataframe(result_df[['title', 'Predicted Rating']])
#
#         fig = px.bar(result_df, x='title', y='Predicted Rating', title='Your Predicted Ratings')
#         st.plotly_chart(fig)
#
# if __name__ == "__main__":
#     main()
