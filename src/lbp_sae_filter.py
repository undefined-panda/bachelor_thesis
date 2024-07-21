import numpy as np
import matplotlib.pyplot as plt
from src.helper_functions import decorate_text
from tqdm import tqdm
import random

def calculate_dynamic_range(image):
    """Calculate the dynamic range of a grayscale image."""
    max_val = np.max(image)
    min_val = np.min(image) if np.min(image) > 0 else 1
    return max_val / min_val

def calculate_entropy(image):
    """Calculate the entropy of a grayscale image."""
    histogram, _ = np.histogram(image, bins=256, range=(0, 256), density=True)
    histogram = histogram[histogram > 0]  # Filter out zero values
    return -np.sum(histogram * np.log2(histogram))

def typical_image_selector(data, labels, num_elements=None):
    data_label, label_count = np.unique(labels, return_counts=True)

    typical_images = {}
    for label in data_label:
        typical_images[label] = []
    
    for label in data_label:
        subset = data[np.where(labels==label)]
        dynamic_ranges = [calculate_dynamic_range(img) for img in subset]
        entropies = [calculate_entropy(img) for img in subset]

        mean_dynamic_range = np.mean(dynamic_ranges)
        mean_entropy = np.mean(entropies)

        typical_images_category = [
            img for img, dr, ent in tqdm(zip(subset, dynamic_ranges, entropies))
            if dr > mean_dynamic_range and ent > mean_entropy
        ]

        typical_images[label] = typical_images_category
        
    decorate_text("Number of typical images selected")
    for i in range(len(label_count)):
        print(f"Label {data_label[i]}: {len(typical_images[data_label[i]])} / {label_count[i]}")

    if num_elements is not None:
        typical_images = {key: random.sample(value, num_elements) for key, value in typical_images.items()}
        print(f"\nRandomly selected {len(typical_images[0])} images per label.")
    
    return typical_images

def get_pixel(img, center, x, y): 
      
    new_value = 0
      
    try: 
        # If local neighbourhood pixel  
        # value is greater than or equal 
        # to center pixel values then  
        # set it to 1 
        if img[x][y] >= center: 
            new_value = 1
              
    except: 
        # Exception is required when  
        # neighbourhood value of a center 
        # pixel value is null i.e. values 
        # present at boundaries. 
        pass
      
    return new_value 

def lbp_calculated_pixel(img, x, y): 
    img = img[0]
   
    center = img[x][y] 
   
    val_ar = [] 
      
    # top_left 
    val_ar.append(get_pixel(img, center, x-1, y-1)) 
      
    # top 
    val_ar.append(get_pixel(img, center, x-1, y)) 
      
    # top_right 
    val_ar.append(get_pixel(img, center, x-1, y + 1)) 
      
    # right 
    val_ar.append(get_pixel(img, center, x, y + 1)) 
      
    # bottom_right 
    val_ar.append(get_pixel(img, center, x + 1, y + 1)) 
      
    # bottom 
    val_ar.append(get_pixel(img, center, x + 1, y)) 
      
    # bottom_left 
    val_ar.append(get_pixel(img, center, x + 1, y-1)) 
      
    # left 
    val_ar.append(get_pixel(img, center, x, y-1)) 
       
    # Now, we need to convert binary 
    # values to decimal 
    power_val = [1, 2, 4, 8, 16, 32, 64, 128] 
   
    val = 0
      
    for i in range(len(val_ar)): 
        val += val_ar[i] * power_val[i] 
          
    return val 

def lbp(image):
    height = image.shape[1]
    width = image.shape[2]
    lbp_image = np.zeros((1, height, width))
    for j in range(height):
        for k in range(width):
            lbp_image[0][j, k] = lbp_calculated_pixel(image, j, k)
    
    return lbp_image

def get_lbp_images(typical_images):
    lbp_images = {category:[] for category in typical_images.keys()}

    for key in typical_images.keys():
        images = typical_images[key]
        for i in range(len(images)):
            img_lbp = lbp(images[i])
            lbp_images[key].append(img_lbp)
    
    return lbp_images