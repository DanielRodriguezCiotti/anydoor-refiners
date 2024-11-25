import streamlit as st
import os
import glob
from pathlib import Path

# Configuration
IMAGE_FOLDER = "dataset/test/cloth"  # Replace with your image folder path
LABEL_FILE = "lora_test_images.txt"  # Path to the text file storing labels

# Function to load images
def load_images(folder):
    return sorted(glob.glob(f"{folder}/*"))

# Function to read existing labels
def read_labels(label_file):
    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            lines = f.readlines()
        return {line.split(",")[0]: line.split(",")[1].strip() for line in lines}
    return {}

# Function to save a label
def save_label(label_file, image_name, label):
    with open(label_file, "a") as f:
        f.write(f"{image_name},{label}\n")

def remove_last_label(label_file):
    with open(label_file, "r") as f:
        lines = f.readlines()
    with open(label_file, "w") as f:
        f.writelines(lines[:-1])

# Main Streamlit app
def main():
    st.title("Image Labeling Tool")
    st.markdown("Label images as `True` or `False`")

    # Load images and labels
    images = load_images(IMAGE_FOLDER)
    labels = read_labels(LABEL_FILE)
    # Count positive and negative labels
    current_count = len(labels)
    true_count = sum([1 for label in labels.values() if label == "True"])
    st.markdown(f"Current count: {true_count}/{current_count} images labeled as `True`")



    # Filter images to show only unlabeled ones
    unlabeled_images = [img for img in images if Path(img).name not in labels]

    if unlabeled_images:
        # Display the first unlabeled image
        current_image = unlabeled_images[0]
        st.image(current_image, caption=Path(current_image).name, width=300)

        # Labeling buttons
        col1, col2, col3 = st.columns(3,gap='small')
        with col1:
            if st.button("True"):
                save_label(LABEL_FILE, Path(current_image).name, "True")
                st.rerun()
        with col2:
            if st.button("False"):
                save_label(LABEL_FILE, Path(current_image).name, "False")
                st.rerun()
        with col3:
            if st.button("Previous"):
                remove_last_label(LABEL_FILE)
                st.rerun()

    else:
        st.success("All images have been labeled!")

if __name__ == "__main__":
    main()
