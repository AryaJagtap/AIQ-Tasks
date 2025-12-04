# some concepts of web scrapping using python with requests and beautifulsoup4
import requests

# before starting web scrapping check the legal rights bu using /robots.txt of any website

web = requests.get("http://127.0.0.1:5500/index.html")

# print(web)
# print(web.status_code)
# print(web.content)
# print(web.url)

#  html parsing

from bs4 import BeautifulSoup

# kinds of objects in beautiful soup
# 1. BeautifulSoup
# 2. Tag
# 3. NavigableString
# 4. Comment

# _______________________________________________________________________________________

soup = BeautifulSoup(web.content, "html.parser")

# print(soup.title)
# print(soup.prettify)

# ______________________________________________________________________________________
# .1 BeautifulSoup

# print(soup.title)
# print(soup.head)

# #  functions in beautiful soup
# print(soup.title) # prints the title tag
# print(soup.title.name) # prints the name of the title tag
# print(soup.title.string) # prints the string inside the title tag
# print(soup.a) # prints the first anchor tag
# print(soup.find_all('a')) # prints all the anchor tags
# print(soup.get_text()) # prints all the text in the html page
# print(soup.find_all('a', href=True)) # prints all the anchor tags with href attribute

# _______________________________________________________________________________________

# 2.Tags in beautiful soup

tag = soup.html
# print(tag)
# print(type(tag))
tag = soup.head
# print(tag)
tag = soup.p
# print(tag)
tag = soup.a
# print(tag)
tag = soup.body
# print(tag)
tag = soup.title
# print(tag)
tag = soup.div
# print(tag)
tag = soup.h1
# print(tag)
tag = soup.h2
# print(tag)
tag = soup.h3
# print(tag).

# _____________________________________________________________________________________
# 3.Navigable STRINGs
#  to get any string inside from any tag , just change the tag and write .string
tag = soup.p.string
print(tag)
print(type(tag)) # it will show that the type is navigable string
# _______________________________________________________________________________________

# 4. Comments

comment = soup.find_all('p')[1].string
print(comment)
print(type(comment)) # it will show that the type is comment
# _______________________________________________________________________________________



# Example of web scrapping
apple = BeautifulSoup(web.content, 'html.parser') # html.parser is used 

# print(apple.prettify()) # prints the html code in a structured way

for link in apple.find_all('a', href=True):
    print(link['href']) # prints the href attribute of each anchor tag
    print(link.get_text()) # prints the text inside each anchor tag