import pandas as pd
import re
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import time

browser = webdriver.Chrome("C:/Users/win11/Desktop/Sem-VI PBL/Power BI/chromedriver.exe")
browser.get("https://www.amazon.in")

search = browser.find_element_by_id("twotabsearchtextbox")
search.send_keys("womens beauty products")
search.send_keys(Keys.RETURN)
time.sleep(2)

# Find all the product elements
products = browser.find_elements_by_xpath("//span[@class='a-size-base-plus a-color-base a-text-normal']")
ratings = browser.find_elements_by_xpath("//span[@class='a-icon-alt']")
People_Reviewed= browser.find_elements_by_xpath("//span[@class='a-size-base s-underline-text']")
Price=browser.find_elements_by_xpath("//span[@class='a-price-whole']")

# Extract and print the names and ratings of the products
for product, rating, t_review,Price in zip(products, ratings, People_Reviewed, Price):
    print("Product:", product.text)
    print("Rating:", rating.get_attribute('textContent'))  # Getting the text content of the rating element
    print("Total Review Count:", t_review.get_attribute("textContent"))
    print("Price:", Price.get_attribute('textContent'))
    print()
