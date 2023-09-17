import os

# Define the directory containing the images
directory_path = 'street_view_image2'
new_path = "C:/Users/Ayberk/Dev/train2"

# Define the prefix you want to remove
prefix_to_remove = 'image_'

# List all files in the directory
file_list = os.listdir(directory_path)

# Iterate through the files and rename them
for filename in file_list:
    if filename.startswith(prefix_to_remove):
        # Construct the new name by removing the prefix
        new_name = filename[len(prefix_to_remove):]
        
        # Full path to the old and new files
        old_file_path = os.path.join(directory_path, filename)
        new_file_path = os.path.join(new_path, new_name)

        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"Renamed {filename} to {new_name}")