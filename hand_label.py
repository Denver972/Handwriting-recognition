import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import pandas as pd

# Function to open an image file


def open_image(image_path):
    image = Image.open(image_path)
    photo = ImageTk.PhotoImage(image)
    return photo

# Function to handle key press and save labels


def on_key_press(event):
    label = event.char
    if label:
        image_labels.append((image_paths[image_index], label))
        print(f"Label '{label}' added for image '{image_paths[image_index]}'.")
        next_image()

# Function to load the next image


def next_image():
    global image_index
    image_index += 1
    if image_index < len(image_paths):
        update_image(image_paths[image_index])
    else:
        print("All images labeled.")
        root.destroy()

# Function to update the displayed image


def update_image(image_path):
    photo = open_image(image_path)
    image_label.config(image=photo)
    image_label.image = photo


# Initialize labels list
image_labels = []

# Read CSV file containing image paths
csv_file_path = filedialog.askopenfilename(
    title="Select CSV file", filetypes=[("CSV files", "*.csv")])
image_df = pd.read_csv(csv_file_path)
image_paths = image_df['image_name'].tolist()

# Create main window
root = tk.Tk()
root.title("Image Labeling")

# Initialize image index
image_index = 0

# Load the first image
photo = open_image(image_paths[image_index])

# Create a label widget to display the image
image_label = tk.Label(root, image=photo)
image_label.pack()

# Bind key press event to the function
root.bind('<Key>', on_key_press)

# Start the GUI event loop
root.mainloop()

# Print the collected labels after the window is closed
print("Collected labels:", image_labels)
