import gc
import os
import csv
import tweepy
import joblib
import pandas as pd
from dotenv import main
from tweet_preprocess import preprocess_tweets

def configure():
    main.load_dotenv()
configure()



#authorize twitter, initialize tweepy
auth = tweepy.OAuthHandler(os.getenv('API_KEY'), os.getenv('API_SECRET_KEY'))
auth.set_access_token(os.getenv('ACCESS_TOKEN'), os.getenv('ACCESS_SECRET_TOKEN'))
api = tweepy.API(auth)

# Load tfidf pipeline


def get_tweets_for_keyword(q, max_tweets=1000):
    """This function gets tweets for a particular keyword"""
    created_at_list = []
    id_str_list = []
    text_list = []
    user_id_list = []
    user_location_list = []
    user_name_list = []
    coordinates_list = []
    retweet_count_list = []

    try:
        for tweet in tweepy.Cursor(api.search_tweets,
                                   q=q,
                                   count=100,
                                   result_type="recent",
                                   include_entities=True,
                                   lang="en").items(max_tweets):

            created_at_list.append(tweet.created_at)
            id_str_list.append(tweet.id_str)
            text_list.append(tweet.text)
            user_id_list.append(tweet.user.id_str)
            user_location_list.append(tweet.user.location)
            user_name_list.append(tweet.user.name)
            coordinates_list.append(tweet.coordinates)
            retweet_count_list.append(tweet.retweet_count)


        # Create a Pandas DataFrame from the data.
        df = pd.DataFrame({'created_at': created_at_list,
                                'id': id_str_list, 
                                'text': text_list, 
                                'user_id': user_id_list, 
                                'user_location': user_location_list, 
                                'user_name': user_name_list, 
                                'coordinates': coordinates_list, 
                                'retweet_count': retweet_count_list})

        tweets_file_name = "tweets_" + q.replace(" ", "_") + ".csv"
        #df.to_csv(f'{tweets_file_name}.csv')
        #print(df)
        print("Saving tweets to file: " + tweets_file_name)
        return df
        
        del df
        gc.collect()
        
    except tweepy.errors.TweepyException as err:
        print(err)

def troll_str_label(troll):
    if troll == "1":
        return "troll"
    else:
        return "not_troll"

def get_troll_pred_tweets(text,max_tweets):
    pipe = joblib.load('sklearn_pipeline.pkl')


# Load saved ml model
    classifier = joblib.load('ml_model.pkl')
    # Get all tweets for the search query.
    _query = text

    # Get tweets for the search query.
    prediction_df = get_tweets_for_keyword(q = _query, max_tweets=max_tweets)
    
    if len(prediction_df) < 1:
        print("No tweets found for the search query.")
        exit()
    
    # Apply text preprocessing.
    prediction_df["processed_text"] = prediction_df["text"].apply(lambda x: preprocess_tweets(x))

    # Predict troll or not. 1=troll, 0=not troll
    prediction_df["predicted_troll"] = classifier.predict(pipe.transform(prediction_df["processed_text"].values))

    # Keep only trolls
    result_df = prediction_df[prediction_df["predicted_troll"] == "1"]
    result_df.reset_index(drop=True, inplace=True)
    result_df["Sl_No"] = result_df.index
    result_df["text"] = result_df["text"].apply(lambda x: x.replace("\n", " ").replace('\r', ''))
    result_df["processed_text"] = result_df["processed_text"].apply(lambda x: x.replace("\r", " ").replace('\n', ''))
    result_df.drop([ "coordinates","id", "retweet_count", "processed_text"], axis=1, inplace=True)
    result_df["predicted_troll"] = result_df["predicted_troll"].apply(lambda x: troll_str_label(x))
    
    return result_df,len(prediction_df)
