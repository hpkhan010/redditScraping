import nltk

# nltk.download()
from collections import Counter

import pprint
import numpy
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import random

import pandas as pd
import praw
import datetime as dt
from IPython import display
import matplotlib.pyplot as plt
import seaborn as sns

# Setup your credentials @ apps.reddit.com
reddit = praw.Reddit(
    client_id=PERSONAL_USE_SCRIPT_14_CHARS,
    client_secret=SECRET_KEY_27_CHARS,
    user_agent=YOUR_APP_NAME,
    username=YOUR_REDDIT_USER_NAME,
    password=YOUR_REDDIT_LOGON_PASSWORD,
)


subreddit = "movies"
keyword = "coronavirus"


# Pulling posts based on keyword search. This allows us to filter posts based
# the keyword we are interested in. We can replace this for loop in the code below.


posts = set()
for submission in reddit.subreddit(subreddit).search(
    keyword, sort="top", time_filter="week", limit=None
):
    posts.add(submission.title)

print(len(posts))


# Pulling top posts regardless of keyword. Same can be done for hot posts.
# "Best" is the highest upvote to downvote ratio, "top" is the most upvotes regardless of downvotes, and "hot" is the most upvotes recently.


posts = {"id": [], "title": [], "body": [], "created": []}

for submission in reddit.subreddit(subreddit).top(time_filter="week", limit=None):
    if not submission.stickied:

        posts["id"].append(submission.id)
        posts["title"].append(submission.title)
        posts["body"].append(submission.selftext)
        posts["created"].append(submission.created)
    # display.clear_output()
print(len(posts["title"]))

df_posts = pd.DataFrame(posts)


def get_date(created):
    return dt.datetime.fromtimestamp(created).strftime("%Y-%m-%d")


timestamp = df_posts["created"].apply(get_date)
df_posts = df_posts.assign(date=timestamp)
del df_posts["created"]
df_posts
# Replace missing values in post body with empty strings
df_posts["body"].fillna(value="", inplace=True)
df_posts.info()

# Adding features to our dataframe.
# Length of title
df_posts["title_length"] = df_posts["title"].apply(lambda x: len(x))

# df_posts.set_index("date", inplace=True)
df_posts.reset_index(inplace=True)
df_posts.head()


dfList = list(set(df_posts["date"]))
dfNames = ["df" + row for row in dfList]

for i, row in enumerate(dfList):
    dfName = dfNames[i]
    dfNew = df_posts[df_posts["date"] == row]
    dfNames[i] = dfNew


tokenizer = nltk.RegexpTokenizer(r"[\w']+")
common_words = []

for df, name in zip(dfNames, dfList):
    all_titles = " ".join([x.lower() for x in df["title"]])
    words = list(tokenizer.tokenize(all_titles))
    words = [x for x in words if x not in stopwords.words("english")]

    common_words.append(words)
    print("Most common words on " + name, "*****", sep="\n")
    pprint.pprint(Counter(words).most_common(5))
    print()
