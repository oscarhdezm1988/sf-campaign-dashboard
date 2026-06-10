"""
SF Campaign Decision Dashboard
==============================
Reads processed_posts_sf.csv (produced by the Colab pipeline) and presents
an interactive dashboard. Deployed on Streamlit Cloud.

The AI brief is optional: if no ANTHROPIC_API_KEY is set in the app's
secrets, the dashboard simply skips it and shows everything else.
"""

import os
import json
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st


CSV_PATH = "processed_posts_sf.csv"   # sits next to this file in the repo
MIN_POSTS = 15                         # volume guard: issues below this are "watch list"
SUMMARY_MODEL = "claude-haiku-4-5-20251001"


# ----------------------------------------------------------------------
# LOAD
# ----------------------------------------------------------------------

st.set_page_config(page_title="SF Campaign Decision Dashboard", layout="wide")

if not os.path.exists(CSV_PATH):
    st.error(f"CSV not found: {CSV_PATH}. Make sure it is in the repo next to this file.")
    st.stop()

df = pd.read_csv(CSV_PATH)
df["created_dt"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
df["issue"] = df["issue_hint"].fillna("other")
df["sentiment_label"] = df["sentiment_label"].fillna("neu")
df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce").fillna(0.0)

# Weights from the pipeline; fall back to flat 1.0 if an older CSV lacks them
if "post_weight" not in df.columns:
    df["post_weight"] = 1.0
df["post_weight"] = pd.to_numeric(df["post_weight"], errors="coerce").fillna(1.0)


# ----------------------------------------------------------------------
# SIDEBAR FILTERS
# ----------------------------------------------------------------------

st.title("SF Campaign Decision Dashboard")
st.caption("Reddit discussion and sentiment, San Francisco / Bay Area")

day_series = df["created_dt"].dt.normalize()
min_day, max_day = day_series.min(), day_series.max()
default_start = max(min_day, max_day - timedelta(days=30))

start, end = st.sidebar.date_input(
    "Date range",
    value=(default_start.date(), max_day.date()),
    min_value=min_day.date(),
    max_value=max_day.date(),
)
subs = st.sidebar.multiselect(
    "Subreddits",
    sorted(df["subreddit"].unique()),
    default=list(sorted(df["subreddit"].unique())),
)
issues_filter = st.sidebar.multiselect("Issues", sorted(df["issue"].unique()), default=None)
sentiments = st.sidebar.multiselect("Sentiment", ["neg", "neu", "pos"],
                                    default=["neg", "neu", "pos"])
min_posts = st.sidebar.slider("Minimum posts for an issue to count as a priority",
                              1, 40, MIN_POSTS)
download_filtered = st.sidebar.checkbox("Limit downloads to current filters", value=True)


# ----------------------------------------------------------------------
# APPLY FILTERS
# ----------------------------------------------------------------------

start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
mask = (
    (day_series >= start_ts) &
    (day_series <= end_ts) &
    (df["subreddit"].isin(subs)) &
    (df["sentiment_label"].isin(sentiments))
)
if issues_filter:
    mask &= df["issue"].isin(issues_filter)

fdf = df.loc[mask].copy()
fdf["date"] = fdf["created_dt"].dt.normalize()


# ----------------------------------------------------------------------
# WEIGHTED PRIORITY with volume guard
# ----------------------------------------------------------------------

def weighted_priority(data):
    if data.empty:
        return pd.DataFrame(columns=["issue", "weighted_mentions", "raw_count",
                                     "weighted_avg_sentiment", "priority"])
    d = data.copy()
    d["w_sent"] = d["post_weight"] * d["sentiment_score"]
    g = d.groupby("issue").agg(
        weighted_mentions=("post_weight", "sum"),
        w_sent_sum=("w_sent", "sum"),
        raw_count=("issue", "size"),
    ).reset_index()
    g["weighted_avg_sentiment"] = g["w_sent_sum"] / g["weighted_mentions"]
    g["priority"] = g["weighted_mentions"] * (0 - g["weighted_avg_sentiment"])
    return g.drop(columns="w_sent_sum")

pri_all = weighted_priority(fdf)
strong = pri_all[pri_all["raw_count"] >= min_posts].sort_values("priority", ascending=False)
thin = pri_all[pri_all["raw_count"] < min_posts].sort_values("priority", ascending=False)


# ----------------------------------------------------------------------
# AI BRIEF (optional, skips cleanly with no key)
# ----------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def generate_narrative(payload_json):
    try:
        import anthropic
    except ImportError:
        return None
    # Reading st.secrets raises if there's no secrets file at all, so guard it.
    key = None
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        key = None
    key = key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    payload = json.loads(payload_json)
    prompt = (
        "You are a political data analyst briefing a San Francisco campaign team. "
        "Using only the data below, write 3 to 5 short sentences in plain English on "
        "which issues are driving the most negative public reaction and why it matters. "
        "Do not invent facts. Avoid jargon and dashes.\n\n"
        f"DATA:\n{json.dumps(payload, indent=2)}"
    )
    try:
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=SUMMARY_MODEL, max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if b.type == "text").strip()
    except Exception as e:
        return f"(brief unavailable: {e})"

st.subheader("Campaign Brief")
if len(fdf):
    payload = {
        "total_posts": int(len(fdf)),
        "avg_sentiment": round(float(fdf["sentiment_score"].mean()), 3),
        "priority_issues": strong.head(6)[
            ["issue", "raw_count", "weighted_avg_sentiment", "priority"]
        ].round(3).to_dict(orient="records"),
    }
    narrative = generate_narrative(json.dumps(payload, sort_keys=True))
    if narrative:
        st.info(narrative)
    else:
        st.caption("Add an ANTHROPIC_API_KEY in the app settings to enable the AI brief. "
                   "Showing charts and tables below.")
else:
    st.info("No data in the selected range.")


# ----------------------------------------------------------------------
# KPIs
# ----------------------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)
c1.metric("Posts", f"{len(fdf):,}")
c2.metric("Avg Sentiment (-1..1)", f"{fdf['sentiment_score'].mean():.3f}" if len(fdf) else "0")
c3.metric("Top Issue", strong.iloc[0]["issue"] if len(strong) else "n/a")
neg_share = (fdf["sentiment_label"].eq("neg").mean() * 100) if len(fdf) else 0
c4.metric("Negative Share", f"{neg_share:.1f}%")

st.write("---")


# ----------------------------------------------------------------------
# CHARTS
# ----------------------------------------------------------------------

st.subheader("Top Issues by Mentions")
top_n = st.slider("Top N", 5, 30, 15, key="top_n")
issue_counts = fdf["issue"].value_counts().head(top_n).reset_index()
issue_counts.columns = ["issue", "mentions"]
if len(issue_counts):
    st.bar_chart(issue_counts.set_index("issue"))

st.subheader("Average Sentiment Over Time")
trend = (fdf.groupby("date")["sentiment_score"].mean()
         .reset_index(name="avg_sentiment").sort_values("date"))
if len(trend):
    st.line_chart(trend.set_index("date"))
else:
    st.info("No data in selected range.")


# ----------------------------------------------------------------------
# PRIORITY TABLES (volume-guarded)
# ----------------------------------------------------------------------

st.subheader(f"Priority Issues (>= {min_posts} posts, trustworthy)")
st.caption("High priority means heavily discussed, recent, and negative. "
           "Negative priority means people are talking about it but not upset.")
st.dataframe(strong.round(3), use_container_width=True)

with st.expander(f"Low-volume issues (< {min_posts} posts, watch list)"):
    st.caption("Too few posts to trust the sentiment, but worth watching for emerging signals.")
    st.dataframe(thin.round(3), use_container_width=True)


# ----------------------------------------------------------------------
# TOP POSTS
# ----------------------------------------------------------------------

st.subheader("Top Posts (by score and comments)")
if len(fdf):
    top_posts = (fdf.sort_values(["score", "num_comments"], ascending=False)
                 .head(25)[["created_dt", "subreddit", "issue", "sentiment_label",
                            "score", "num_comments", "title", "permalink"]])
    st.dataframe(top_posts, use_container_width=True)


# ----------------------------------------------------------------------
# DOWNLOAD
# ----------------------------------------------------------------------

st.write("---")
dl_df = fdf if download_filtered else df
st.download_button("Download CSV", dl_df.to_csv(index=False).encode("utf-8"),
                   file_name="sf_filtered_posts.csv", mime="text/csv")
