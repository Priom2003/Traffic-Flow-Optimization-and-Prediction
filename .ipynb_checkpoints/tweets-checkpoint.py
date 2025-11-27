import requests

url = "https://twitter241.p.rapidapi.com/getTweets"

querystring = {"query":"traffic jam near Bangalore", "count":"50"}

headers = {
    "X-RapidAPI-Key": "a3b7c9e9d6mshaaa55032a385fd9p1d429djsn893fae372d5f",
    "X-RapidAPI-Host": "twitter241.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)
data = response.json()

for tweet in data['tweets']:
    print(tweet['text'])
