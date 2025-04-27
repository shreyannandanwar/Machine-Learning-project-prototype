# 🎬 Personalized Movie Recommender

A lightweight, interactive **Movie Recommender System** built with **Streamlit**, **Scikit-learn**, and **Plotly**.  
It predicts personalized movie recommendations based on a few movie ratings you provide, using a **Decision Tree Regressor** inside a **MultiOutputRegressor**.

---

## 🚀 Features

- Rate a few random movies and get personalized recommendations
- Visualize the top 10 recommendations in an interactive Plotly chart
- Peek into how the Decision Tree model predicts (Tree Visualization)
- Randomized movie sampling to make the experience fresh on each run
- Powered by a fast and explainable ML model: Decision Trees

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend / ML**: Scikit-learn (DecisionTreeRegressor, MultiOutputRegressor)
- **Visualization**: Matplotlib, Plotly
- **Data**: [MovieLens Latest Small Dataset (ml-latest-small)](https://grouplens.org/datasets/movielens/latest/)

---

## 📂 Project Structure

```plaintext
├── Personalized_Movie_Recommender.py
├── ml-latest-small/
│   ├── ratings.csv
│   ├── movies.csv
├── README.md
├── requirements.txt
```

---

## 📥 Setup Instructions

1. **Clone this repository**
   ```bash
   git clone https://github.com/shreyannandanwar/your-repo-name.git
   cd Movie-suggester
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download MovieLens Data**
   - Download [ml-latest-small.zip](https://grouplens.org/datasets/movielens/latest/)
   - Extract and place the `ratings.csv` and `movies.csv` inside the `ml-latest-small/` directory.

4. **Run the app**
   ```bash
   streamlit run Personalized_Movie_Recommender.py
   ```

5. **Enjoy your movie recommendations!** 🎬 🍿

---

## 📈 How It Works

1. Randomly sample movies from the MovieLens dataset.
2. Ask the user to rate a few movies using Streamlit sliders.
3. Train a **MultiOutput Decision Tree Regressor**:
   - Inputs: your given ratings
   - Outputs: predicted ratings for unseen movies
4. Recommend the top movies based on predicted ratings.
5. Optionally, visualize the internal Decision Tree structure.

---

## 📸 Demo Screenshots

| Step | Screenshot |
| :--- | :--------- |
| Rate Movies | ![Rate Movies](assets/rate_movies.png) |
| Top Recommendations | ![Recommendations](assets/recommendations.png) |
| Decision Tree Visualization | ![Decision Tree](assets/decision_tree.png) |

(*Note: You can add screenshots after running the app!*)

---

## 🧹 To-Do Improvements

- Use more sophisticated models (Random Forest, LightGBM)
- Add user login for saving preferences
- Add genres filtering (Action, Comedy, Drama, etc.)
- Deploy online (Streamlit Cloud / HuggingFace Spaces)

---

## 🤝 Contributing

Pull requests are welcome!  
For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the MIT License.

---

## 📬 Contact

- **Author**: [Shreyan Nandanwar](https://github.com/shreyannandanwar)
- **Email**: shreyannandanwar@gmail.com

---

# ✨ Happy Recommending!

---

---

Would you also like me to generate a ready `requirements.txt` file for you? 🚀  
It'll make setting up super easy!
