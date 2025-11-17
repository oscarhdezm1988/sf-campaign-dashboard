import os
import pandas as pd
import numpy as np
import streamlit as st
from datetime import timedelta

# ---------- Load ----------
CSV_PATH = "/content/processed_posts_sf.csv"  # change if needed
if not os.path.exists(CSV_PATH):
    st.set_page_config(page_title="SF Campaign Decision Dashboard", layout="wide")
    st.error(f"CSV not found: {CSV_PATH}. Make sure you've created it first.")
    st.stop()

df = pd.read_csv(CSV_PATH)

# Parse times and basic fields
df["created_dt"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
df["issue"] = df["issue_hint"].fillna("other")
df["sentiment_label"] = df["sentiment_label"].fillna("neu")
df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce").fillna(0.0)

# ---------- UI ----------
st.set_page_config(page_title="SF Campaign Decision Dashboard", layout="wide")
st.title("SF Campaign Decision Dashboard")
st.caption("Reddit discussion & sentiment — San Francisco / Bay Area")

# Sidebar filters — work purely with normalized Timestamps
day_series = df["created_dt"].dt.normalize()
min_day = day_series.min()
max_day = day_series.max()

# Default: last 30 days (bounded by available range)
default_start = max(min_day, max_day - timedelta(days=30))
default_end = max_day

start, end = st.sidebar.date_input(
    "Date range",
    value=(default_start.date(), default_end.date()),
    min_value=min_day.date(),
    max_value=max_day.date()
)

subs = st.sidebar.multiselect(
    "Subreddits",
    sorted(df["subreddit"].unique()),
    default=list(sorted(df["subreddit"].unique()))
)
issues = st.sidebar.multiselect("Issues", sorted(df["issue"].unique()), default=None)
sentiments = st.sidebar.multiselect("Sentiment", ["neg","neu","pos"], default=["neg","neu","pos"])
st.sidebar.write("---")
download_filtered = st.sidebar.checkbox("Limit table downloads to current filters", value=True)

# ---------- Filtering (Timestamp ↔ Timestamp) ----------
start_ts = pd.Timestamp(start)
end_ts   = pd.Timestamp(end)
mask = (
    (day_series >= start_ts) &
    (day_series <= end_ts) &
    (df["subreddit"].isin(subs)) &
    (df["sentiment_label"].isin(sentiments))
)
if issues:
    mask &= df["issue"].isin(issues)
fdf = df.loc[mask].copy()
fdf["date"] = fdf["created_dt"].dt.normalize()  # for grouping/plotting

# ---------- KPIs ----------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Posts", f"{len(fdf):,}")
avg_sent = fdf["sentiment_score"].mean() if len(fdf) else 0
col2.metric("Avg Sentiment (−1..1)", f"{avg_sent:.3f}")
top_issue = fdf["issue"].value_counts().idxmax() if len(fdf) else "—"
col3.metric("Top Issue (by mentions)", top_issue)
neg_share = (fdf["sentiment_label"].eq("neg").mean()*100) if len(fdf) else 0
col4.metric("Negative Share", f"{neg_share:.1f}%")

st.write("---")

# ---------- Charts ----------
# Top issues by mentions
st.subheader("Top Issues by Mentions")
top_n = st.slider("Top N", 5, 30, 15, key="top_n")
issue_counts = fdf["issue"].value_counts().head(top_n).reset_index()
issue_counts.columns = ["issue", "mentions"]
st.bar_chart(issue_counts.set_index("issue"))

# Sentiment trend (daily avg)
st.subheader("Average Sentiment Over Time")
trend = (
    fdf.groupby("date")["sentiment_score"]
    .mean()
    .reset_index(name="avg_sentiment")
    .sort_values("date")
)
if len(trend):
    st.line_chart(trend.set_index("date"))
else:
    st.info("No data in selected range.")

# ---------- Focus table (volume x negativity) ----------
st.subheader("🧭 Suggested Focus (High Volume × Negative Sentiment)")
vol = fdf["issue"].value_counts()
score = fdf.groupby("issue")["sentiment_score"].mean()
focus = (vol.to_frame("mentions")
         .join(score.to_frame("avg_sentiment"))
         .assign(priority=lambda x: x["mentions"] * (0 - x["avg_sentiment"]))
         .sort_values("priority", ascending=False)
         .head(10)
         .reset_index().rename(columns={"index":"issue"}))
st.dataframe(focus, use_container_width=True)

# ---------- “Top posts” table ----------
st.subheader("Top Posts (by score & comments)")
def top_posts(data, n=25):
    return (data.sort_values(["score","num_comments"], ascending=False)
                .head(n)[["created_dt","subreddit","issue","sentiment_label","score","num_comments","title","permalink"]])

st.dataframe(top_posts(fdf), use_container_width=True)

# ---------- Insights summary ----------
st.subheader("📊 Summary Insights")
fdf["week"] = fdf["created_dt"].dt.to_period("W").dt.start_time
wk = fdf.groupby(["week","issue"]).size().reset_index(name="mentions")
if len(wk["week"].unique()) >= 2:
    latest = wk["week"].max()
    prev = sorted(wk["week"].unique())[-2]
    cur = wk[wk["week"]==latest].set_index("issue")["mentions"]
    pre = wk[wk["week"]==prev].set_index("issue")["mentions"]
    change = (cur - pre).fillna(cur).sort_values(ascending=False)
    risers = ", ".join(change.head(3).index.tolist())
    fallers = ", ".join(change.tail(3).index.tolist())
else:
    risers = fallers = "—"

issue_sent = fdf.groupby("issue")["sentiment_score"].mean().sort_values()
most_neg = ", ".join(issue_sent.head(3).index.tolist()) if len(issue_sent) else "—"
most_pos = ", ".join(issue_sent.tail(3).index.tolist()) if len(issue_sent) else "—"

st.write(f"**Rising issues:** {risers}")
st.write(f"**Falling issues:** {fallers}")
st.write(f"**Most negative:** {most_neg}")
st.write(f"**Most positive:** {most_pos}")

# ---------- Downloads ----------
st.write("---")
dl_df = fdf.copy() if download_filtered else df.copy()
st.download_button("Download CSV", dl_df.to_csv(index=False).encode("utf-8"),
                   file_name="sf_filtered_posts.csv", mime="text/csv")
