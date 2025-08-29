from http.client import TOO_EARLY
from multiprocessing import process
from operator import contains
from tracemalloc import stop
import aiml
from jupyterlab_server import translator
import numpy
from pkg_resources import split_sections
from regex import P
import wikipedia 
import os
import sys
import time
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk import RegexpParser
from nltk.sem import Expression
from nltk.inference import ResolutionProver
from APIFootball import APIFootball 

import nltk.corpus


model = tf.keras.models.load_model("top3Leageue_classifier.h5")
class_names = ["La Liga", "Serie A", "Premier League"]

def predict_logo(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128)) 
    img_array = tf.keras.preprocessing.image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = img_array / 255.0  
    

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]  
    
    return predicted_class


#load API team Ids
api = APIFootball()
api.getTeamIds()

#initialise knowledge base
read_expr = Expression.fromstring
kb=[]
data = pd.read_csv('logic-kb.csv', header=None)
[kb.append(read_expr(row)) for row in data[0]]


def checkContradiction(expr):
    prover = ResolutionProver()
    try:
        return prover.prove(nltk.sem.Expression.negate(expr), kb)
    except Exception as e:
        print(f"Contradiction or unknown error occurred: {str(e)} ")
        return False

def remove_articles(text):
    articles = {'a', 'an', 'the'}
    words = text.split()
    return ' '.join(word for word in words if word.lower() not in articles)

#load QAPairs
df = pd.read_csv('QAPairs.csv')
question = df['Question'].tolist()

#pre-process QAPairs
stopwords = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.WordNetLemmatizer()
remove_punctuation = str.maketrans('', '', string.punctuation)
processed_question = []

for q in question:

    q = q.lower().translate(remove_punctuation)
    tokens = nltk.word_tokenize(q)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords ]
    processed_question.append(" ".join(tokens))


#vectorize QAPairs
vectorizer = TfidfVectorizer(lowercase= True, stop_words='english')
tfidfMatrix = vectorizer.fit_transform(processed_question)


#Create Kernel Object
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="mybot-basic.xml")

# Welcome user
print("Welcome to this chat bot. Please feel free to ask questions from me!")

# Main loop
while True:
    #get user input
    try:
        userInput = input("> ")
    except (KeyboardInterrupt, EOFError):
        print("Bye!")
        break
    #pre-process user input and determine response agent (if needed)

    inputVector = vectorizer.transform([userInput])

    #calculate cosine similarity
    cosineSimilarity = cosine_similarity(inputVector, tfidfMatrix)

#activate selected response agent
    if cosineSimilarity[0,cosineSimilarity.argmax()] > 0.7:
        responseAgent = 'csv'
    else: 
        responseAgent = 'aiml'

#generate response
    if responseAgent == 'csv':
        answer = df['Answer'][cosineSimilarity.argmax()]
       
    if responseAgent == 'aiml':
        answer = kern.respond(userInput)
    #post-process the answer for commands
    if answer[0] == '#': 
         params = answer[1:].split('$')
         cmd = int(params[0]) 
         if cmd == 0:
            print(params[1])
            break
         #Wikipedia
         elif cmd == 1:
            try:
                wpage = wikipedia.summary(params[1], sentences=3, auto_suggest=True)
                print(wpage)
            except wikipedia.exceptions.PageError:
                print("I can't find anything for this online, specific the full name")
            except wikipedia.exceptions.DisambiguationError as e:
                print(f"Multiple matches found. Be more specfifc. Options:  {e.options}")

            
            #APIFootball

         elif cmd == 2:
             a = params[1].split('*')
             topic = a[0]
             team = a[1]
            # print(topic)
             
             match topic:
                    case "standings":
                        print(api.getStandings())
                    case "teams":
                        print(api.getTeams())
                    case "team":
                        print(api.getTeam(team))
                    case "match":
                        print(api.getMatches(team,status="last"))
                    case "matches":
                        print(api.getMatches(team))
                    case "scorers":
                        print(api.getScorers(team))
                    case "coach":
                        print(api.getCoach(team))
        #Knowledge Base
         elif cmd == 31:
            try: 

                object, subject=params[1].split(' is ')
                object = object.replace(" ", "_")
                remove_articles(subject).strip()

                expr=read_expr(f"{subject}({object})")
                print(expr)
                if expr in kb:
                    print("I already know that")
                    continue
                elif checkContradiction(expr):
                    print("Error adding fact KB, contradiction detected")
                else:
                    kb.append(expr)
                    print('OK, I will remember that',object,'is', subject)
                  
            except nltk.sem.logic.LogicalExpressionException:
                print('I could not understand that. Please try again.')
        #KB
         elif cmd == 32:
            try:

                object,subject=params[1].split(' is ')
                object = object.replace(" ", "_")
                remove_articles(subject).strip()
                expr=read_expr(subject + '(' + object + ')')
                
                if ResolutionProver().prove(expr,kb):
                    print('Correct.')
                else:
                    print('I could not find any information on that online or in my knowledge base. This is most likely false.')
            except nltk.sem.logic.LogicalExpressionException:
                print("I could not understand that. Please try again")
        
         elif cmd == 33: 
            try:
                image_path = params[1]  
                prediction = predict_logo(image_path)  
                print(f"I think this logo belongs to: {prediction}")
            except Exception as e:
                print(f"Error processing the image: {str(e)}")
         elif cmd == 99:
            print("I did not get that, please try again.")
    
    else:
        print(answer)
