import requests
from bs4 import BeautifulSoup
from googletrans import Translator #pip install googletrans==3.1.0a0 fix error
response = requests.get("https://www.worldometers.info/coronavirus/country/viet-nam/#graph-cases-daily")

soup = BeautifulSoup(response.content, "html.parser")
translate = Translator()
time = str(soup.find_all('div', attrs={"style": "font-size:13px; color:#999; text-align:center"}))
date = translate.translate(time.split(">")[1].split("<")[0], dest='vi',src="en").text
#cases = str(soup.findAll('div', class_='maincounter-number'))
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

import tkinter as tk

root = tk.Tk()
root.geometry("300x200")

def func(event):
    print("You hit return.")

def onclick(event=None):
    print("You clicked the button")

root.bind('<Return>', onclick)

button = tk.Button(root, text="click me", command=onclick)
button.pack()

root.mainloop()

