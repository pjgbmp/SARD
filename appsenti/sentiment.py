from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Se cargan UNA sola vez
vader = SentimentIntensityAnalyzer()

roberta = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
)

LABEL_MAP = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive",
    "negative": "Negative",
    "neutral": "Neutral",
    "positive": "Positive"
}


def vader_sentiment(text):
    score = vader.polarity_scores(text)["compound"]

    if score >= 0.05:
        return "Positive", score
    elif score <= -0.05:
        return "Negative", score
    else:
        return "Neutral", score


def roberta_sentiment(text):
    result = roberta(text[:512])[0]
    label = LABEL_MAP[result["label"]]
    score = result["score"]
    return label, score


def hybrid_sentiment(title, content=""):
    title_label, title_score = vader_sentiment(title)
    content_label, content_score = vader_sentiment(content[:500])

    vader_score = (title_score * 0.6) + (content_score * 0.4)
    vader_label = (
        title_label if abs(title_score) >= abs(content_score)
        else content_label
    )   

    combined_text = title + ". " + content[:600]  #ver si agregar analisis por chunks de texto
    rob_label, rob_score = roberta_sentiment(combined_text)

    if vader_label == rob_label:
        final_label = rob_label
        confidence = (abs(vader_score) + rob_score) / 2
    else:
        final_label = rob_label
        confidence = rob_score * 0.7

    return {
        "sentiment": final_label,
        "confidence": round(confidence, 3),
        "vader": vader_label,
        "roberta": rob_label
    }


def analyze_items(items):
    results = []

    for item in items:
        s = hybrid_sentiment(item["text"], "")
        s["source"] = item["source"]
        s["engagement"] = item["engagement"]
        results.append(s)

    return results


def aggregate_by_source(results):
    summary = {}

    for r in results:
        src = r["source"]
        score = (
            1 if r["sentiment"] == "Positive"
            else -1 if r["sentiment"] == "Negative"
            else 0
        )

        summary.setdefault(src, []).append(score)

    return {
        src: sum(vals) / len(vals)
        for src, vals in summary.items()
        if vals
    }


def aggregate_global(results):
    scores = [
        1 if r["sentiment"] == "Positive"
        else -1 if r["sentiment"] == "Negative"
        else 0
        for r in results
    ]
    return sum(scores) / len(scores) if scores else 0
