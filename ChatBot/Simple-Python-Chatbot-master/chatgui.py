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
import tkinter
from tkinter import *

response = requests.get("https://www.worldometers.info/coronavirus/country/viet-nam/#graph-cases-daily")
soup = BeautifulSoup(response.content, "html.parser")
translate = Translator()
time = str(soup.find_all('div', attrs={"style": "font-size:13px; color:#999; text-align:center"}))
date = translate.translate(time.split(">")[1].split("<")[0], dest='vi',src="en").text

lemmatizer = WordNetLemmatizer()
model = load_model('D:/Chatbot/Simple-Python-Chatbot-master/chatbot_model2.h5', compile=False)
intents = json.loads(open('./Simple-Python-Chatbot-master/intents.json',encoding='utf8' ).read())
words = pickle.load(open('./Simple-Python-Chatbot-master/words.pkl','rb'))
classes = pickle.load(open('./Simple-Python-Chatbot-master/classes.pkl','rb'))

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

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

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

#Creating GUI with tkinter

def send(a =None):
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "Bạn: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

        res = chatbot_response(msg)
        if res == "Tình hình dịch Covid-19 ở Việt Nam: ":
            ChatLog.insert(END, "Bot: " + res + total() + '\n\n')
        elif res == "Hôm nay Việt Nam ghi nhận: ":
            ChatLog.insert(END, "Bot: " + res + today() + '\n\n')
        else:
            ChatLog.insert(END, "Bot: " + res + '\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


base = Tk()
base.title("Hello")
base.geometry("500x500")
base.resizable(width=FALSE, height=FALSE)
base.configure(background='#7EEBE6')
#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)

ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Gửi", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send)

#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
base.bind('<Return>', send)
#Place all components on the screen
scrollbar.place(x=476,y=6, height=386)
ChatLog.place(x=6,y=6, height= 386, width=470)
EntryBox.place(x=20, y=401, height=90, width=265)
SendButton.place(x=350, y=401, height=90)

base.mainloop()
