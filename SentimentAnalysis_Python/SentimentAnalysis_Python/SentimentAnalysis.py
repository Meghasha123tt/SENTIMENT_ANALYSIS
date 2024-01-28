
import speech_recognition as sr
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')


def sentiment_analysis(text):
    
    tokens = word_tokenize(text)
   
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
  
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(' '.join(filtered_tokens))
   
    if scores['compound'] >= 0.05:
        label = 'positive'
    elif scores['compound'] <= -0.05:
        label = 'negative'
    else:
        label = 'neutral'
    return label


def analyze_voice_sentiment():
    """Analyze the sentiment of speech input and return a sentiment label."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!") 
        audio = recognizer.listen(source)
    try:
       
        text = recognizer.recognize_google(audio)
        print("Recognized Text:", text) 
        label = sentiment_analysis(text)
        print("Sentiment Label:", label) 
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio") 
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e)) 


print("Press Enter next to the \">\" to start and type \"exit\" to end the chat.")
while True:
    user_input = input("> ")
    if user_input.lower() == "exit":
        print("Chat ended.")
        break
    else:
        analyze_voice_sentiment()


