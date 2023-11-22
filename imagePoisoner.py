import numpy as np
import os

def add_poison_to_images(dataset_path, poison_image_path, poison_percentage):
    # Load poison image
    poison_image = np.fromfile(poison_image_path, dtype=np.uint8)
    
    # Iterate through each image file in the dataset
    for filename in os.listdir(dataset_path):
        if filename.endswith(".bin"):
            image_path = os.path.join(dataset_path, filename)
            
            # Load image
            image = np.fromfile(image_path, dtype=np.uint8)
            
            # Calculate the number of pixels to poison
            num_pixels = int(len(image) * poison_percentage)
            
            # Replace pixels with poison image pixels
            image[:num_pixels] = poison_image[:num_pixels]
            
            # Apply adversarial perturbation to the poisoned pixels
            perturbation = np.random.randint(0, 256, size=num_pixels, dtype=np.uint8)
            image[:num_pixels] = np.clip(image[:num_pixels] + perturbation, 0, 255)
            
            # Save the poisoned image
            image.tofile(image_path)
