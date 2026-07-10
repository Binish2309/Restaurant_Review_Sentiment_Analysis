# 🍽️ Restaurant Review Sentiment Analysis using NLP

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/NLTK-NLP-4CAF50?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Streamlit-WebApp-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

<p align="center">
  A production-quality NLP project that analyses restaurant reviews and classifies them as
  <strong>Positive</strong>, <strong>Negative</strong>, or <strong>Neutral</strong> — surfacing
  actionable business insights for restaurant owners.
</p>

---


## 🎯 Objectives

1. **Clean & preprocess** raw restaurant review text using a 7-step NLP pipeline  
2. **Explore the data** with rich visualisations (ratings, word clouds, trends)  
3. **Classify sentiment** using VADER + TextBlob (no training required)  
4. **Train an ML model** (TF-IDF + Logistic Regression) as a production artifact  
5. **Extract business insights** and translate them into actionable recommendations  
6. **Deploy a Streamlit app** for real-time review analysis  

---

## 🛠️ Technologies Used

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Plotly, WordCloud |
| NLP | NLTK, TextBlob, VADER |
| Machine Learning | Scikit-learn (TF-IDF, Logistic Regression) |
| Web App | Streamlit |
| Serialisation | Joblib |

---

## 📂 Dataset

The project ships with a synthetic but realistic dataset of **1 000 restaurant reviews** modelled
after the publicly available [Yelp Dataset](https://www.yelp.com/dataset) and
[Restaurant Reviews on Kaggle](https://www.kaggle.com/datasets/vigneshwarsofficial/reviews).

**Fields:**

| Column | Description |
|--------|-------------|
| `reviewer_id` | Anonymous reviewer identifier |
| `restaurant_name` | Name of the restaurant |
| `review` | Raw review text |
| `rating` | Star rating (1–5) |
| `date` | Review date (2022–2024) |
| `true_sentiment` | Ground-truth label (for evaluation) |

**Why this dataset?**
- Covers all three sentiment classes with realistic class imbalance (55 % positive, 30 % negative, 15 % neutral)
- Mirrors the schema of real-world Yelp / Google Reviews data
- No PII or licensing restrictions

---

## 📁 Folder Structure

```
Restaurant_Review_Sentiment_Analysis/
│
├── notebook.ipynb              ← Complete Jupyter Notebook (EDA + Modelling)
├── app.py                      ← Streamlit web application
├── build_notebook.py           ← Script that generates notebook.ipynb
│
├── dataset/
│   ├── generate_dataset.py     ← Synthetic dataset generator
│   └── restaurant_reviews.csv  ← Generated dataset (after running generator)
│
├── images/                     ← All saved visualisations
│   ├── rating_distribution.png
│   ├── review_length.png
│   ├── top_words.png
│   ├── wordclouds.png
│   ├── sentiment_distribution.png
│   ├── vader_compound.png
│   ├── rating_vs_sentiment.png
│   └── confusion_matrix.png
│
├── models/
│   └── sentiment_pipeline.pkl  ← Trained TF-IDF + LogReg pipeline
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10 or higher  
- pip package manager  
- (Optional) virtualenv or conda

### Step 1 — Clone the Repository
```bash
git clone https://github.com/Binish2309/Restaurant-Review-Sentiment-Analysis.git
cd Restaurant-Review-Sentiment-Analysis
```

### Step 2 — Create a Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Generate the Dataset
```bash
python dataset/generate_dataset.py
```

### Step 5 — Open the Notebook
```bash
jupyter notebook notebook.ipynb
```

### Step 6 — Launch the Streamlit App
```bash
streamlit run app.py
```
The app opens at **http://localhost:8501** in your browser.

---

## 🔄 Project Workflow

```
Raw Reviews
    │
    ▼
Data Cleaning ──────────────── Handle missing, remove duplicates
    │
    ▼
Text Preprocessing ──────────── Lowercase → Remove noise → Tokenise → Lemmatise
    │
    ▼
EDA ─────────────────────────── Ratings, lengths, word frequency, temporal trends
    │
    ▼
Sentiment Analysis ──────────── VADER (primary) + TextBlob (cross-validation)
    │
    ▼
ML Classification ───────────── TF-IDF + Logistic Regression → saved model
    │
    ▼
Business Insights ───────────── Actionable recommendations from data patterns
    │
    ▼
Streamlit App ───────────────── Real-time inference with prediction history
```

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Total Reviews | 1 000 |
| VADER ↔ TextBlob Agreement | ~85 % |
| ML Classifier Accuracy | ~88 % |
| Positive Reviews | ~55 % |
| Negative Reviews | ~30 % |
| Neutral Reviews | ~15 % |

### Key Insights
- **Wait times** and **cold food** are the top negative drivers  
- **Food quality** and **staff friendliness** drive 5-star reviews  
- Negative reviews are on average **40 % longer** than positive ones  
- Restaurants with >60 % positive reviews correlate with 4.2+ average rating  

---

## 🚀 Future Improvements

- [ ] Fine-tune **BERT / RoBERTa** for domain-specific accuracy  
- [ ] Implement **aspect-based sentiment** (food, service, ambiance as separate scores)  
- [ ] Connect to **Yelp Fusion API** or **Google Places API** for live data  
- [ ] Add **multilingual support** using `langdetect` + multilingual BERT  
- [ ] Build a **business dashboard** with restaurant comparison and trend alerts  
- [ ] Add **topic modelling** (LDA / BERTopic) to discover latent review themes  

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Binish Mohammad Asif Gandhi**  
[![Email](https://img.shields.io/badge/Email-gandhibinish%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:gandhibinish@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Binish%20Gandhi-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/binish-gandhi-42b54722b/)
[![GitHub](https://img.shields.io/badge/GitHub-Binish2309-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Binish2309)
---

<p align="center">⭐ Star this repo if you found it helpful!</p>
