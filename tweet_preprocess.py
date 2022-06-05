import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import re
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import itertools
from emoji import demojize

# Initialize the tweet tokenizer
tokenizer = TweetTokenizer()

# Initialize the stopwords
stops = set(stopwords.words("english"))

# Initialize the text processor from ekphrasis
text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'url', 'date', 'number'],
    # terms that will be annotated
    #annotate={"hashtag", "allcaps", "elongated", "repeated",
    #    'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens
    
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector="twitter", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=True,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)



def preprocess_tweets(s):
    s = " ".join(text_processor.pre_process_doc(s))
    s = ' '.join(k for k, _ in itertools.groupby(s.split()))
    s = s.replace("' ", "'").replace(" '", "'")
    s = " ".join(s.split())
    s = s.strip()
    s = s.replace('“', "")
    
    s = " ".join([demojize(token) for token in tokenizer.tokenize(s)])
    s = s.replace("_", " ")
    s = [re.sub('[^A-Za-z0-9]+', '', word) for word in tokenizer.tokenize(s) if word not in stops]
    s = " ".join(s).split()
    s = " ".join(s)
    #s = s.replace("<user>", '')
    return s
    
# Some example tweets and their processed tweets printed.
#sentences = [
#    "CANT WAIT for the new season of #TwinPeaks ＼(^o^)／!!! #davidlynch #tvseries :)))",
#    "I saw the new #johndoe movie and it suuuuucks!!! WAISTED $10... #badmovies :/",
#    "@SentimentSymp:  can't wait for the Nov 9 #Sentiment talks!  YAAAAAAY !!! :-D http://sentimentsymposium.com/.",
#    "'You have a #problem? Yes! Can you do #something about it? No! Than why  '",
#    "on the bright side , my music theory teacher just pocket dabbed and said ,'i know what's hip .'and walked away 😂 😭"
#]

#print("\n")
#for i in sentences:
#    print(preprocess_tweets(i))