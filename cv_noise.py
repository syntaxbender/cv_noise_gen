import cv2
import numpy as np
from noise import pnoise2
import random


height=400
width=650
image = np.zeros((height, width, 4), dtype=np.uint8)
image[:, :, 3] = 0

num_spots = 1
min_radius = 150
max_radius = 200

min_blur_x = 201
max_blur_x = 301
min_blur_y = 41
max_blur_y = 101
threshold = 0.2

def random_odd_number(min_value, max_value):
    min_odd = min_value if min_value % 2 == 1 else min_value + 1
    max_odd = max_value if max_value % 2 == 1 else max_value - 1

    if min_odd > max_odd:
        raise ValueError("No odd numbers available in the given range")

    return random.choice(range(min_odd, max_odd + 1, 2))

def generate_perlin_noise(width, height, scale=50):
    """Perlin Noise kullanarak büyük ölçekli bloklar halinde noise haritası üretir."""
    noise_map = np.zeros((height, width), dtype=np.float32)
    
    for y in range(height):
        for x in range(width):
            noise_map[y, x] = pnoise2(x / scale, y / scale, octaves=3)
    
    noise_map = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min())
    return noise_map

def generate_circle(height,width,min_blur_x, max_blur_x, min_blur_y, max_blur_y,min_radius,max_radius):
    temp_img = np.zeros((height, width), dtype=np.float32)
    x = np.random.randint(0, width)
    y = np.random.randint(0, height)
    blur_x = random_odd_number(min_blur_x, max_blur_x)
    blur_y = random_odd_number(min_blur_y, max_blur_y)
    radius = np.random.randint(min_radius, max_radius)
    cv2.circle(temp_img, (x, y), radius, 255, -1)
    temp_img = cv2.GaussianBlur(temp_img, (blur_x, blur_y), 0)
    return temp_img / 255

def open_image(image):
    image = cv2.imread(image, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
    return image
  
def apply_perlin_to_alpha(image, scale=50, min_alpha=0.6, max_alpha=1.0):
    if image.shape[2] == 3: 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    h, w, _ = image.shape
    noise_map = generate_perlin_noise(w, h, scale)
    
    img_alpha_channel = image[:, :, 3] 
    mask = img_alpha_channel > 0 
    alpha_variation = noise_map 

    circle = circles_layer(height=h,width=w,min_blur_x=min_blur_x,max_blur_x=max_blur_x,min_blur_y=min_blur_y,max_blur_y=max_blur_y,min_radius=min_radius,max_radius=max_radius)
    
    # cv2.imwrite("aa.jpg", (circle_mask*255).astype(np.uint8))    
    # cv2.imwrite("bb.jpg", (circle_mask2*255).astype(np.uint8))

    alpha_variation[mask] = 1-((circle[mask]) * alpha_variation[mask])
    img_alpha_channel[mask] = (img_alpha_channel[mask] * alpha_variation[mask])

    image[:, :, 3] = img_alpha_channel
    return image
    
def circles_layer(height,width,min_blur_x, max_blur_x, min_blur_y, max_blur_y,min_radius,max_radius):
  image = np.zeros((height, width), dtype=np.float32)
  for _ in range(num_spots):
    temp_img = generate_circle(
      height=height,
      width=width,
      min_blur_x=min_blur_x,
      max_blur_x=max_blur_x,
      min_blur_y=min_blur_y,
      max_blur_y=max_blur_y,
      min_radius=min_radius,
      max_radius=max_radius
    )
    image = np.where(temp_img > 0, temp_img, 0) + np.where(image > 0, image, 0)
  return image

watermark = open_image("../image.png")
watermark_alpha_noise = (apply_perlin_to_alpha(watermark, scale=80)*255).astype(np.uint8)
cv2.imwrite("watermark_with_perlin_alpha.png", watermark_alpha_noise)
