import nltk

# nltk.download()

import config
import pandas as pd
import praw
import datetime as dt
from IPython import display
import matplotlib.pyplot as plt
import seaborn as sns


reddit = praw.Reddit(
    client_id=config.PERSONAL_USE_SCRIPT_14_CHARS,
    client_secret=config.SECRET_KEY_27_CHARS,
    user_agent=config.YOUR_APP_NAME,
    username=config.YOUR_REDDIT_USER_NAME,
    password=config.YOUR_REDDIT_LOGON_PASSWORD,
)

# pulling a sample of posts
posts = {"title": [], "created": []}

subreddit = "movies"
keyword = "coronavirus"

for submission in reddit.subreddit(subreddit).search(
    keyword, sort="top", time_filter="week", limit=None
):
    posts["title"].append(submission.title)
    posts["created"].append(submission.created)
    display.clear_output()
    print(len(posts))


from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import pprint

sia = SIA()
results = []

for line in posts["title"]:
    pol_score = sia.polarity_scores(line)
    pol_score["Post"] = line
    results.append(pol_score)


print(results)

df = pd.DataFrame.from_records(results)
df.head()

# labelling dataset
df["label"] = 0
df.loc[df["compound"] >= 0.2, "label"] = 1
df.loc[df["compound"] < 0.2, "label"] = -1
df.head()

# printing sample of positive posts
print(list(df[df["label"] == 1].Post)[:5])

# printing sample of negative posts
print(list(df[df["label"] == -1].Post)[:5])

# Word Distributions
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords

tokenizer = RegexpTokenizer(r"\w+")
stop_words = stopwords.words("english")


def process_text(posts):
    tokens = []
    for line in posts:
        toks = tokenizer.tokenize(line)
        toks = [t.lower() for t in toks if t.lower() not in stop_words]
        tokens.extend(toks)

    return tokens


# Positive Words

pos_lines = list(df[df.label == 1].Post)

pos_tokens = process_text(pos_lines)
pos_freq = nltk.FreqDist(pos_tokens)

pos_freq.most_common(20)

# Plotting positive word frequency
y_val = [x[1] for x in pos_freq.most_common()]

fig = plt.figure(figsize=(10, 5))
plt.plot(y_val)

plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Word Frequency Distribution (Positive)")
plt.show()

# Negative Words

neg_lines = list(df[df.label == -1].Post)

neg_tokens = process_text(neg_lines)
neg_freq = nltk.FreqDist(neg_tokens)

neg_freq.most_common(20)

# Plotting negative word frequency
y_val = [x[1] for x in neg_freq.most_common()]

fig = plt.figure(figsize=(10, 5))
plt.plot(y_val)

plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Word Frequency Distribution (Negative)")
plt.show()


# Pulling sample of comments
