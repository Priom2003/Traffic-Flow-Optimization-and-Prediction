import requests
import json
import os
import time
import csv
import re
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd

# 🔑 Load environment variables
load_dotenv()
API_KEY = os.environ.get("RAPIDAPI_KEY")

if not API_KEY:
    raise ValueError("API key not found. Please set RAPIDAPI_KEY in your .env file.")

# --- Configuration ---
queries_to_search = [
    "Vellore old bus stand traffic", "Vellore new bus stand jam update",
    "Vellore market road traffic", "Vellore town bus route delay",
    "Vellore signal free corridor update",
    "Chennai Vellore highway traffic update", "Tirupati Vellore road jam",
    "Arcot Vellore highway congestion", "Vellore Salem road traffic",
    "Vellore Krishnagiri highway traffic",
    "Vellore flyover construction traffic", "Katpadi overbridge delay update",
    "Vellore underground drainage road block",
    "Vellore smart city project traffic impact", "Road widening near VIT Vellore",
    "Vellore heavy rain traffic block", "Vellore storm water drainage issue",
    "Vellore rainy season road jam", "Flooded roads in Vellore update",
    "Vellore waterlogging near CMC",
    "Vellore ambulance stuck in traffic", "CMC emergency road block Vellore",
    "Katpadi hospital traffic update", "Vellore patient transfer delayed due to jam",
    "Vellore town bus late update", "Vellore mofussil bus stand traffic",
    "Katpadi railway station traffic jam", "Vellore train delay due to road block",
    "Vellore auto stand congestion",
    "Vellore traffic fine update", "Vellore traffic police checking",
    "Katpadi no entry traffic", "Vellore one way traffic violation",
    "Road closed by Vellore police",
    
    # "Vellore traffic", "Vellore traffic jam", "Vellore accident", "Vellore road closed",
    # "Vellore signal jam", "Vellore police traffic control", "Katpadi traffic",
    # "Katpadi accident", "Katpadi railway station road", "CMC Vellore traffic",
    # "CMC road closed", "Gandhinagar Vellore traffic", "Bagayam Vellore accident",
    # "Arcot road traffic", "Bengaluru highway Vellore", "NH48 Vellore traffic",
    # "Vellore heavy rain road", "Vellore flood road", "Vellore waterlogging",
    # "Vellore storm damage", "Vellore temple festival traffic",
    # "Katpadi college event traffic", "VIT Vellore traffic",
    # "Vellore traffic congestion", "Vellore road accident", "Vellore highway jam",
    # "Vellore road repair work", "Katpadi road traffic", "Katpadi bus stand traffic",
    # "VIT campus road jam", "CMC hospital road traffic", "Gandhinagar road accident",
    # "Arcot road accident", "NH48 traffic delay", "Vellore monsoon traffic",
    # "Vellore festival crowd", "Vellore bus stand jam", "Long queue Vellore toll",
    # "Bengaluru highway accident", "Katpadi overbridge traffic",
    # "Vellore bridge repair traffic", "Vellore bypass road closed",
    # "Katpadi flood road block", "CMC gate traffic jam", "Katpadi market crowd",
    # "Vellore two-wheeler accident", "Vellore truck accident",
    # "NH48 vehicle breakdown Vellore", "Vellore traffic updates",
    # "Vellore signal failure", "Vellore one way traffic", "Vellore road diversion",
    # "Katpadi traffic police", "Katpadi road widening", "CMC emergency traffic",
    # "Gandhinagar traffic signal", "Arcot road bottleneck",
    # "Vellore bus strike traffic", "Vellore lorry breakdown",
    # "NH48 Vellore toll jam", "Bengaluru highway slow traffic Vellore",
    # "Katpadi station traffic update", "Vellore rain traffic jam",
    # "CMC campus road jam", "Bagayam hospital traffic",
    # "Long queue at Katpadi crossing", "Vellore school bus traffic",
    # "Katpadi railway crossing closed", "Vellore festival traffic block",
    # "Arcot road heavy vehicle jam", "Vellore road accident update",
    # "Vellore bike accident news", "Katpadi road closed today"
]

url = "https://twitter241.p.rapidapi.com/search"
headers = {
    "x-rapidapi-key": API_KEY,
    "x-rapidapi-host": "twitter241.p.rapidapi.com"
}

output_csv = "fetch_tweets.csv"
MIN_TWEETS_PER_QUERY = 100
# --- End Configuration ---

def extract_location_from_text(text):
    """Try to detect location names in tweet text"""
    known_places = [
        "Vellore", "Katpadi", "CMC Hospital", "CMC Vellore",
        "Gandhinagar", "Bagayam", "Arcot", "Bengaluru highway", "NH48"
    ]
    for place in known_places:
        if re.search(rf"\b{place}\b", text, re.IGNORECASE):
            return place
    return None

def fetch_min_tweets(query, min_tweets=100):
    collected_tweets = []
    cursor = None

    while len(collected_tweets) < min_tweets:
        querystring = {"type": "Latest", "count": "20", "query": query}
        if cursor:
            querystring["cursor"] = cursor

        try:
            response = requests.get(url, headers=headers, params=querystring)
            response.raise_for_status()
            data = response.json()

            entries = data.get('result', {}).get('timeline', {}).get('instructions', [{}])[0].get('entries', [])
            new_tweets_found = 0
            next_cursor = None

            for entry in entries:
                content = entry.get('content', {})

                # Tweets
                if content.get('entryType') == 'TimelineTimelineItem':
                    tweet_result = content.get('itemContent', {}).get('tweet_results', {}).get('result', {})
                    if tweet_result:
                        tweet_id = str(tweet_result.get('rest_id'))
                        created_at_raw = tweet_result.get('legacy', {}).get('created_at')
                        full_text = tweet_result.get('legacy', {}).get('full_text', "").strip()

                        # Format datetime
                        try:
                            dt_obj = datetime.strptime(created_at_raw, "%a %b %d %H:%M:%S %z %Y")
                            created_at = dt_obj.strftime("%d %b %Y %I:%M:%S %p")
                        except:
                            created_at = created_at_raw

                        # Location
                        user_info = tweet_result.get('core', {}).get('user_results', {}).get('result', {}).get('legacy', {})
                        user_location = user_info.get("location") or extract_location_from_text(full_text)

                        retweet_count = tweet_result.get('legacy', {}).get('retweet_count', 0)
                        like_count = tweet_result.get('legacy', {}).get('favorite_count', 0)

                        collected_tweets.append({
                            "tweet_id": tweet_id,
                            "created_at": created_at,
                            "raw_text_tweet": full_text,
                            "user_location": user_location or "",
                            "retweet_count": int(retweet_count),
                            "like_count": int(like_count)
                        })
                        new_tweets_found += 1

                # Cursor
                if content.get('entryType') == 'TimelineTimelineCursor' and content.get('cursorType') == 'Bottom':
                    next_cursor = content.get('value')

            print(f"  > {new_tweets_found} new tweets, total: {len(collected_tweets)} for '{query}'")

            if not next_cursor or new_tweets_found == 0:
                break

            cursor = next_cursor
            time.sleep(1)

        except Exception as e:
            print(f"Error for query '{query}': {e}")
            break

    return collected_tweets

def main():
    # Load existing dataset if exists
    if os.path.exists(output_csv):
        df_existing = pd.read_csv(output_csv, dtype=str)
    else:
        df_existing = pd.DataFrame(columns=["tweet_id", "created_at", "raw_text_tweet", "user_location", "retweet_count", "like_count"])

    all_tweets_data = []

    for query in queries_to_search:
        print(f"\n🔍 Fetching tweets for: {query}")
        tweets = fetch_min_tweets(query, min_tweets=MIN_TWEETS_PER_QUERY)
        all_tweets_data.extend(tweets)
        print("-" * 20)
        time.sleep(1)

    if not all_tweets_data:
        print("No tweets collected.")
        return

    # Append new data to existing dataframe
    df_new = pd.DataFrame(all_tweets_data)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)

    # Remove duplicates based on tweet_id
    df_combined.drop_duplicates(subset=["tweet_id"], inplace=True)

    # Save updated dataset
    df_combined.to_csv(output_csv,mode = 'a',header=not os.path.exists(output_csv), index=False, encoding='utf-8')
    print(f"\n✅ Dataset updated: {len(df_combined)} unique tweets saved in '{output_csv}'")

if __name__ == "__main__":
    main()
