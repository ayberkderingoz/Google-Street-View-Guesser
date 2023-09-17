import os
import csv
import requests

# Define your Google Street View API key
api_key = 'YOUR_API_KEY'

# Define the directory where you want to save the street view images
output_directory = 'street_view_image2'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Define the starting index for naming
starting_index = 33624

# Initialize counters for both successful downloads and total processed rows
success_index = starting_index
total_processed = 0

# Define the CSV file paths
csv_file_input = 'random_coord_dataset2.csv'
csv_file_output = 'succesful_coords2.csv'

# Open the input CSV file for reading and the output CSV file for writing
with open(csv_file_input, mode='r') as input_file, \
     open(csv_file_output, mode='w', newline='') as output_file:

    csv_reader = csv.DictReader(input_file)
    csv_writer = csv.writer(output_file)

    # Write the header row to the output CSV file
    csv_writer.writerow(['Latitude', 'Longitude'])

    for idx, row in enumerate(csv_reader, start=starting_index):
        # Get latitude and longitude from the CSV row
        latitude = row['Latitude']
        longitude = row['Longitude']

        # Define the Google Street View Image API query URL
        query_url = f"https://maps.googleapis.com/maps/api/streetview?size=640x640&location={latitude},{longitude}&key={api_key}"

        # Send a GET request to retrieve the Street View image
        image_response = requests.get(query_url)

        if image_response.status_code == 200:
            # Save the image with a filename based on the success index
            image_file_name = os.path.join(output_directory, f"image_{success_index}.png")
            with open(image_file_name, 'wb') as image_file:
                image_file.write(image_response.content)
            print(f"Saved {image_file_name}")
            # Write the successful coordinates to the output CSV file
            csv_writer.writerow([latitude, longitude])
            output_file.flush()
            # Increment the success index
            success_index += 1
        else:
            print(f"Failed to retrieve image for coordinates {latitude},{longitude}")

        # Increment the total processed rows
        total_processed += 1

print(f"Downloaded {success_index - starting_index} out of {total_processed} images.")