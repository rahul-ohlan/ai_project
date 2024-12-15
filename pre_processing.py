import os
import pandas as pd

# Step 1: Get unique file names from the specified directory without the '.npz' extension
directory_path = "/home/raid/home/rohlan/cloome/data"

# List all files in the directory
all_files = os.listdir(directory_path)
print("All files in the directory:", all_files)

# Filter files ending with '.npz'
unique_files = [os.path.splitext(file)[0] for file in all_files if file.endswith('.npz')]
print("Unique files (without .npz):", unique_files)

# Step 2: Read the CSV file and extract the sample-plate column
csv_file_path = "image_embeddings.csv"
image_embeddings = pd.read_csv(csv_file_path)

# Extract the sample-plate names directly from the CSV (assuming it's the first column)
sample_plate_column = image_embeddings.columns[0]  # Adjust if the column is not the first one
sample_plate_names = image_embeddings[sample_plate_column].unique()

# Step 3: Find the intersection between directory file names and the sample-plate names
intersection = set(unique_files).intersection(sample_plate_names)

# Output the results
print("Unique files from directory:", len(unique_files))
print("Sample plate names from CSV:", len(sample_plate_names))
print("Intersection count:", len(intersection))

# Step 4: Filter the original CSV based on the intersection
filtered_data = image_embeddings[image_embeddings[sample_plate_column].isin(intersection)]

# Step 5: Save the filtered data to a new CSV file
filtered_csv_file_path = "filtered_image_embeddings.csv"
filtered_data.to_csv(filtered_csv_file_path, index=False)
print(f"Filtered data saved to {filtered_csv_file_path}")
