# 18-9-23
# CSC461 – Assignment1 – Web Scraping
# Ayesha Zahid
# Fa21-BSE-003
# Create a web scraper in Python to extract the ‘title’ and ‘rating’ for 5 of your
# most favorite movies from the IMDB website.
import requests
from bs4 import BeautifulSoup
import time
import pandas as pd

#  URL for IMDb
base_url = "https://www.imdb.com"

#  URLs of your 5 favorite movies
favorite_movies = [
    "https://www.imdb.com/title/tt0111161/",  # The Shawshank Redemption
    "https://www.imdb.com/title/tt0468569/",  # The Dark Knight
    "https://www.imdb.com/title/tt1375666/",  # Inception
    "https://www.imdb.com/title/tt0110912/",  # Pulp Fiction
    "https://www.imdb.com/title/tt0137523/"   # Fight Club
]

# Initialize lists to store movie data
titles = []
ratings = []

# Iterate through the list of movie URLs
for movie_url in favorite_movies:
    # Send a GET request to the movie URL with headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }
    response = requests.get(movie_url, headers=headers)  # Include headers in the request

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract the title
        title = soup.find("h1").text.strip()

        # Extract the rating if available, or set it to None if not found
        try:
            rating = float(soup.find("span", itemprop="ratingValue").text)
        except AttributeError:
            rating = None

        # Append the data to the respective lists
        titles.append(title)
        ratings.append(rating)

        # Sleep for 1 second to avoid overloading IMDb's servers
        time.sleep(1)
    else:
        print(f"Failed to fetch data for {movie_url}")
        title = soup.find("h1").text.strip()

# Extract the rating if available, or set it to None if not found
rating_element = soup.find("div", class_="ratingValue")
if rating_element:
    rating = float(rating_element.find("span", itemprop="ratingValue").text)
else:
    rating = None

# Create a DataFrame and export data to an Excel file
data = {"Title": titles, "Rating": ratings}
df = pd.DataFrame(data)

# Save the DataFrame to an Excel file 
df.to_excel("favorite_movies.xlsx", index=False)

# Display the DataFrame
print(df)
