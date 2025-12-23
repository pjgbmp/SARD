import requests
from bs4 import BeautifulSoup
import snscrape.modules.twitter as sntwitter
import praw

# ---------- NOTICIAS ----------

def fetch_article_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        return " ".join(p.get_text() for p in paragraphs).strip()
    except Exception:
        return ""

# ---------- TWITTER / X (GRATIS) ----------

def fetch_twitter_posts(query, limit=30):
    results = []
    for i, tweet in enumerate(
        sntwitter.TwitterSearchScraper(query).get_items()
    ):
        if i >= limit:
            break

        results.append({
            "source": "twitter",
            "text": tweet.content,
            "engagement": tweet.likeCount + tweet.retweetCount
        })

    return results

# ---------- REDDIT (GRATIS) ----------

#reddit = praw.Reddit(
#    client_id="TU_CLIENT_ID",
#    client_secret="TU_CLIENT_SECRET",
#    user_agent="sentiment_app"
#)

#def fetch_reddit_posts(query, limit=20):
#    results = []
#
#    for submission in reddit.subreddit("all").search(query, limit=limit):
#        text = submission.title + ". " + (submission.selftext or "")
#        results.append({
#            "source": "reddit",
#            "text": text,
#            "engagement": submission.score
#        })
#
#    return results
#