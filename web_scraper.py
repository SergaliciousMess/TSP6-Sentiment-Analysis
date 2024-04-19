import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By


# Get request

req = requests.get('https://www.rogerebert.com/reviews/the-last-airbender-2010') 

# Parsing HTML
soup = BeautifulSoup(req.content, 'html.parser')

# print(soup.title)

# Class is class of text, content stored under "p" tag
bodyText = soup.find('div', class_='page-content container is-fullhd ads--reviewPage')
content = bodyText.find_all('p')

# For loop removing tags from text
for word in content:
    print(word.text)

# Stars are stored as images, many stars on the page
# top_star finds the stars stored in the specific div (these are the ones for the review)
top_star = soup.find('div', class_='page-content--star-rating')
num_star = 0

# If the page actually has a star rating, count the number of stars in the above class
if top_star:
    stars = top_star.find_all('i', class_='icon-star-full')
    num_star = len(stars)

print("This page has", num_star, "stars")


# # Automate Firefox
# drive = webdriver.Firefox()

# # Get Ebert Reviews
# drive.get("https://www.rogerebert.com/reviews")

# dropmenu = Select(drive.find_element(By.ID, 'star_ratings_from'))

# #dropmenu.select_by_visible_text('★★★★')

# dropmenu.select_by_value('4.0')
