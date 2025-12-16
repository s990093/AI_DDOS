import cv2
import os
import glob
import numpy as np
import math

def load_images(image_dir, max_images=100):
    img_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp')
    image_paths = []
    for ext in img_extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    
    image_paths = sorted(image_paths)[:max_images]  # 限制最多圖片數（避免 OOM）
    
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
    return images

def resize_images(images, size):
    return [cv2.resize(img, size) for img in images]

def make_grid(images, grid_size):
    rows, cols = grid_size
    h, w, _ = images[0].shape
    grid_img = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)

    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        if r < rows:
            grid_img[r*h:(r+1)*h, c*w:(c+1)*w] = img
    return grid_img

def calculate_grid_size(n):
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols

def show_images_grid_fullscreen(image_dir):
    images = load_images(image_dir)
    if not images:
        print("❌ 沒有讀到任何圖片！")
        return
    # 將圖片 resize 成一致大小
    resized = resize_images(images, (300, 300))  # 每張圖 300x300，可依需要調整
    
    # 計算排版格子數
    grid_size = calculate_grid_size(len(resized))
    
    # 製作拼圖大圖
    grid_img = make_grid(resized, grid_size)
    
    # 顯示拼圖圖像（全螢幕）
    cv2.namedWindow("Image Grid", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Image Grid", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Image Grid", grid_img)

    print("🖼️ 所有圖片已拼接顯示（按任意鍵關閉）")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ✅ 設定你的圖片資料夾路徑
image_folder = "your/image/folder/path"
show_images_grid_fullscreen(image_folder)
