#Importing libraries
import sys
import os
import jsonpickle
import tweepy



consumer_key = 'VBo9qa8ftO61mhSaYxHd6XHSn'

consumer_secret = '9IqWfyO8TPeo4xj8fSaKMsJji5LbQh6JZWvOBbNpxvPLDFTOem'





access_token = '2828468661-MRhcCGtBTOiBiRJlL0x4q6torL1WMm6leZjuaVH'

access_secret = '7SgCleNBDZfWPGQNmKlmWNF2DX3wXaYFCVJ2JCdtqUiR9'


#Pass our consumer key and consumer secret to Tweepy's user authentication handler
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

#Pass our access token and access secret to Tweepy's user authentication handler
auth.set_access_token(access_token, access_secret)

#Creating a twitter API wrapper using tweepy
#Details here http://docs.tweepy.org/en/v3.5.0/api.html
api = tweepy.API(auth)

#Error handling
if (not api):
    print ("Problem connecting to API")

print('Enter granularity ("city", "state", "country")')
granularity = input()

print('Enter Location')
location = input()
#Getting Geo ID
places = api.geo_search(query=location, granularity=granularity)

print('Enter key phrase to search')
keyphrase = input()
#Copy id
place_id = places[0].id

#Switching to application authentication
auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)

#Setting up new api wrapper, using authentication only
api = tweepy.API(auth, wait_on_rate_limit=True,wait_on_rate_limit_notify=True)

#Error handling
if (not api):
    print ("Problem Connecting to API")
print('a')
searchQuery = 'place:' + place_id + keyphrase
#Maximum number of tweets we want to collect
maxTweets = 10000
#The twitter Search API allows up to 100 tweets per query
tweetsPerQry = 100

tweetCount = 0

#Open a text file to save the tweets to
with open('CollectedTweets.json', 'w') as f:

    #Tell the Cursor method that we want to use the Search API (api.search)
    #Also tell Cursor our query, and the maximum number of tweets to return
    for tweet in tweepy.Cursor(api.search,q=searchQuery).items(maxTweets) :

        #Verify the tweet has place info before writing (It should, if it got past our place filter)
        #if tweet.place is not None and tweet.place.name == location:
        print(tweet.text)
        print('b')
        #Write the JSON format to the text file, and add one to the number of tweets we've collected
        f.write(jsonpickle.encode(tweet._json, unpicklable=False) + '\n')
        tweetCount += 1
    print('c')
    #Display how many tweets we have collected
    print("Downloaded {0} tweets".format(tweetCount))
