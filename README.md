# Movie Recommender System 🎬🤖

An end-to-end Machine Learning web application that recommends movies based on content similarity. The system analyzes movie metadata (genres, cast, crew, and keywords) to suggest the top 5 most relevant films to the user.

## ✨ Features
- **Search Functionality:** Select a movie from a dropdown of 5,000+ titles.
- **Accurate Recommendations:** Uses Cosine Similarity to find mathematical matches.
- **Interactive UI:** Built with Streamlit for a smooth user experience.
- **Fast Processing:** Pre-computed similarity matrices stored as Pickle files for instant results.

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, NLTK
- **Frontend:** Streamlit
- **Model Storage:** Pickle

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/mkhizarsid/Movie-Recommender-system.git](https://github.com/mkhizarsid/Movie-Recommender-system.git)
   pip install -r requirements.txt
   streamlit run app.py
