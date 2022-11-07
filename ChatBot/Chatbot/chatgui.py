# -*- coding: utf-8 -*-
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import requests
from bs4 import BeautifulSoup
from googletrans import Translator #pip install googletrans==3.1.0a0 fix error
import json
import random

response = requests.get("https://www.worldometers.info/coronavirus/country/viet-nam/#graph-cases-daily")
soup = BeautifulSoup(response.content, "html.parser")
translate = Translator()
time = str(soup.find_all('div', attrs={"style": "font-size:13px; color:#999; text-align:center"}))
date = translate.translate(time.split(">")[1].split("<")[0], dest='vi',src="en").text

lemmatizer = WordNetLemmatizer()
model = load_model('D:/ChatBot/Chatbot/chatbot_model2.h5')
intents = json.loads(open('D:/ChatBot/Chatbot/intents.json',encoding='utf8' ).read())
words = pickle.load(open('D:/ChatBot/Chatbot/words.pkl','rb'))
classes = pickle.load(open('D:/ChatBot/Chatbot/classes.pkl','rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def total():
    cases = str(soup.find_all('div', attrs={"class": "maincounter-number"}))
    case = cases.split(">")[2].split(" <")[0]
    dead = cases.split(">")[6].split("<")[0]
    recovered = cases.split(">")[10].split("<")[0]
    texttotal = f"\nTổng số ca nhiễm: {case}\nTổng số người mất: {dead}\nTổng số người được chữa khỏi: {recovered}\n{date}\n"
    return texttotal

def today():
    new = str(soup.findAll('li', class_='news_li'))
    new_case = new.split(">")[2].split(" ")[0]
    new_dead = new.split(">")[4].split(" ")[0]
    texttoday = f"\nSố ca nhiễm: {new_case}\nSố người mất: {new_dead}\n{date}\n"
    return texttoday

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    if res == "Tình hình dịch Covid-19 ở Việt Nam: ":
        res = res + total()
    elif res == "Hôm nay Việt Nam ghi nhận: ":
        res = res + today()
    else:
        return res
    return res


text = "hello"
print(chatbot_response(text))
