SF Campaign Decision Dashboard
A real-time political analytics dashboard for monitoring San Francisco voter sentiment using Reddit + NLP.

This project is an end-to-end political data pipeline and interactive dashboard that tracks public sentiment, trending issues, and discussion patterns across key Bay Area subreddits. It is designed to help political campaigns, policy teams, and data analysts understand what voters care about right now.

Built with Python, Streamlit, and NLP (RoBERTa sentiment model), the dashboard allows anyone to explore the political landscape of San Francisco through data.

🚀 Live App

👉 https://YOUR-APP-LINK-HERE.streamlit.app

(Add this after deployment)

🔍 Features
✔ Reddit Data Pipeline

Pulls live posts from targeted subreddits (SanFrancisco, BayArea, AskSF, etc.)

Filters content using place-based + issue-based keyword matching

Extracts metadata (title, score, comments, timestamps)

✔ NLP Sentiment Analysis

Uses RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest)

Generates:

Sentiment score (−1 to +1)

Sentiment label (neg/neu/pos)

✔ Issue Classification

Automatically tags posts with major SF political topics:

Housing

Crime

Homelessness

Transit (MUNI/BART)

Immigration

Rent / cost of living

Local elections

and more...

✔ Interactive Analytics Dashboard

Hosted on Streamlit Cloud (no backend server required).

Includes visualizations such as:

Top Issues by Mentions

Sentiment Trends Over Time

Suggested Campaign Focus (volume × negativity)

Top Posts (highest score + comments)

Download filtered datasets

🛠️ Tech Stack

Languages & Libraries

Python

Pandas / NumPy

PRAW (Reddit API)

Transformers (HuggingFace RoBERTa sentiment model)

Streamlit

Altair

Tools

Google Colab (pipeline execution)

GitHub

Streamlit Cloud (app hosting)

📁 Repository Structure
sf-campaign-dashboard/
│
├── streamlit_app.py        # Dashboard UI + charts
├── processed_posts_sf.csv  # Processed dataset with sentiment
├── requirements.txt        # Dependencies for Streamlit Cloud
└── README.md               # Documentation (this file)

⚙️ How the Pipeline Works

Connect to Reddit API using PRAW

Download posts from selected subreddits

Filter posts using SF neighborhoods + political issue keywords

Clean & preprocess text

Run RoBERTa sentiment model

Save results to CSV

Dashboard loads the CSV for visualization

📈 Example Insights This Dashboard Produces

Which issues are spiking week-over-week

Which topics have the most negative sentiment

Neighborhood-specific complaints

Momentum leading into elections

What policies or events trigger sentiment swings

These insights are useful for:

Political campaigns

Policy researchers

Journalists

Civic tech groups

Public opinion analysts

💼 About the Developer

Oscar Hernandez
Data Scientist & Political Data Analyst & NLP Developer
Specialized in machine learning, sentiment modeling, and political behavior.


📬 Contact

If you'd like to collaborate, connect, or request the full pipeline:

📧 Email: oscarhdezm0825@gmail.com
💼 LinkedIn: https://www.linkedin.com/in/oscarhernandezmata/

⭐ Support

If you find this useful, please consider starring the repo!
