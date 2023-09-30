# 18-9-23
# CSC461 – Assignment1 – Web Scraping
# Ayesha Zahid
# Fa21-BSE-003
# Create a web scraper in Python to extract the ‘title’ and ‘rating’ for 5 of your
# most favorite movies from the IMDB website.


import requests
from bs4 import BeautifulSoup

#  URL for 'timeanddate' website
timeanddate_url = "https://www.timeanddate.com/on-this-day/november/23"

# the URL for 'britannica' website
britannica_url = "https://www.britannica.com/on-this-day/November-23"

# function to scrape and extract information about notable individuals
def scrape_birthdate_info(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
       
        birthdate_section = soup.find("div", class_="col__content")
        if birthdate_section:
            # Extract the list of names
            names = [li.text.strip() for li in birthdate_section.find_all("li")]
            return names
        else:
            return []
    else:
        return []

#function to scrape and extract historical events
def scrape_events(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        # Find the list of events, if available
        events_list = soup.find("ul", class_="activity__list")
        if events_list:
            # Extract the list of events
            events = [li.text.strip() for li in events_list.find_all("li")]
            return events
        else:
            return []
    else:
        return []

# Extract information from 'timeanddate' website (notable individuals)
birthdate_names = scrape_birthdate_info(timeanddate_url)

# Extract information from 'britannica' website (historical events)
britannica_events = scrape_events(britannica_url)

# Printing the information
print("1) My birth date is 23 November, and I share my birthdate with:")
if birthdate_names:
    for name in birthdate_names:
        print(f"- {name}")
else:
    print("- No notable individuals found on this day.")

print("\n2) On my birthdate (November 23rd), the following events happened:")
if britannica_events:
    for event in britannica_events:
        print(f"- {event}")
else:
    print("- No historical events found on this day.")


