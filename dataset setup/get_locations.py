import osmnx as ox
import random
import csv
from shapely.geometry import Point, Polygon
from geopy.geocoders import Nominatim

# Define the bounding box (geographical area) for your region of interest
# Adjust these coordinates to specify the area you want to generate random coordinates within
north, south, east, west = 60, -60, 180, -130  # Example bounding box for New York City

# Create a Polygon representing the geographical boundaries
boundary_polygon = Point(west, south).buffer(1).envelope.union(Point(east, north).buffer(1).envelope)

# Number of random coordinates to generate and save to the CSV file
num_coordinates = 24000

# Initialize a geocoder
geolocator = Nominatim(user_agent="geoapi")

# Generate and store random coordinates on land in a list
random_coordinates = []
while len(random_coordinates) < num_coordinates:
    random_lat = random.uniform(south, north)
    random_lon = random.uniform(west, east)
    random_coordinates.append((random_lat,random_lon))


# Define the CSV file path
csv_file_path = "random_coordinates.csv"

# Save the coordinates to the CSV file
with open(csv_file_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    
    # Write the header row (optional)
    writer.writerow(["Latitude", "Longitude"])
    
    # Write the random coordinates to the CSV file
    for lat, lon in random_coordinates:
        writer.writerow([lat, lon])

print(f"Random coordinates on land saved to {csv_file_path}")