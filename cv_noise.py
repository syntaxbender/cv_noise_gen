import cv2
import numpy as np
from noise import pnoise2
import random


height=400
width=650
image = np.zeros((height, width, 4), dtype=np.uint8)
image[:, :, 3] = 0

num_spots = 10
min_radius = 10
max_radius = 70

#color = (0,0,0,255)
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
    
    # Normalize (-1,1) değerlerini (0,255) aralığına çek
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
def normalize_array(arr, new_min, new_max):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    return new_min + (arr - arr_min) * (new_max - new_min) / (arr_max - arr_min)
  
def apply_perlin_to_alpha(image, scale=50, min_alpha=0.6, max_alpha=1.0):
    """Perlin Noise ile alpha kanalında bloklu (düzensiz) şeffaflık oluşturur."""
    if image.shape[2] == 3:  # Eğer alpha kanalı yoksa ekle
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    h, w, _ = image.shape
    # Perlin Noise haritası oluştur
    noise_map = generate_perlin_noise(w, h, scale)
    
    # Alpha kanalını al
    img_alpha_channel = image[:, :, 3]  # 0-1 aralığında normalize et
    cv2.imwrite("xx.jpg", (img_alpha_channel*255).astype(np.uint8))
    
    # Sadece alpha > 0 olan yerlere noise uygula (arka plan tamamen saydam olan yerlere etki etmez)
    mask = img_alpha_channel > 0.5  # Sadece görünür pikseller için işlem yap
    cv2.imwrite("xxd.jpg", (mask*255).astype(np.uint8))
    
    #alpha_variation = (noise_map / 255) * test  # Alpha değişkenliği oluştur
    alpha_variation = noise_map  # Alpha değişkenliği oluştur
    # alpha_variation = np.where(noise_map < 0.3, 0.0, alpha_variation)
    # alpha_variation = np.where(noise_map < 0.6, 0.2, alpha_variation)
    # alpha_variation = np.where(noise_map < 0.9, 0.4, alpha_variation)
    # alpha_variation = np.where(noise_map < 0.8, 0.4, alpha_variation)
    #alpha_variation = alpha_variation > 1;
    
    circle = circles_layer(height=h,width=w,min_blur_x=min_blur_x,max_blur_x=max_blur_x,min_blur_y=min_blur_y,max_blur_y=max_blur_y,min_radius=min_radius,max_radius=max_radius)
    
    circle_mask = circle > 0  # Sadece görünür pikseller için işlem yap
    circle_mask2 = circle <= 0  # Sadece görünür pikseller için işlem yap
    np.savetxt("dosya.txt", alpha_variation, fmt="%.6f")  # 6 basamak hassasiyetle kaydeder
    
    
    
    # np.savetxt("dosya.txt", noise_map, fmt="%.6f")  # 6 basamak hassasiyetle kaydeder
    # np.savetxt("dosya2.txt", alpha_variation, fmt="%.6f")  # 6 basamak hassasiyetle kaydeder
    alpha_variation[circle_mask] = (circle[circle_mask] * alpha_variation[circle_mask])
    alpha_variation = normalize_array(alpha_variation*0.001,0,1)
    
    # np.savetxt("dosya.txt", alpha_variation, fmt="%.6f")  # 6 basamak hassasiyetle kaydeder
    
    # np.savetxt("dosya2.txt", alpha_variation, fmt="%.6f")  # 6 basamak hassasiyetle kaydeder
    
    alpha_variation[circle_mask2] = 0
    img_alpha_channel[mask] = 1-(img_alpha_channel[mask] * alpha_variation[mask])

    # Güncellenmiş alpha kanalını uygula
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

    #image = cv2.addWeighted(image, 1, temp_img, 1, 0)
  return image
# Görseli yükle (PNG formatında olmalı)

watermark = open_image("../image2.png")

# Alpha kanalına sadece Perlin Noise uygula

watermark_alpha_noise = cv2.GaussianBlur((apply_perlin_to_alpha(watermark, scale=80)*255).astype(np.uint8), (1, 1), 0)

# Sonucu kaydet
cv2.imwrite("watermark_with_perlin_alpha.png", watermark_alpha_noise)
