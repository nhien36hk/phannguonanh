import cv2
import numpy as np
from tkinter import Tk, Button, Label, filedialog, Frame, StringVar
from PIL import Image, ImageTk

# Hàm chọn ảnh đầu vào từ người dùng
def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            original_image.set(file_path)
            display_original_image(img)

# Hàm hiển thị ảnh gốc
def display_original_image(img):
    resized_img = cv2.resize(img, (300, 300))  # Thay đổi kích thước ảnh gốc
    original_img = ImageTk.PhotoImage(image=Image.fromarray(resized_img))
    original_label.config(image=original_img)
    original_label.image = original_img

# Hàm tính ngưỡng Otsu và các giá trị m1, m2, global mean
def otsu_threshold(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    total_pixels = image.size
    p = hist / total_pixels

    mg = np.dot(np.arange(256), p)

    max_between_class_variance = 0
    threshold_value = 0
    m1 = m2 = 0
    weight_A = 0
    sum_A = 0

    for k in range(256):
        weight_A += p[k]
        if weight_A == 0:
            continue

        weight_B = 1 - weight_A
        if weight_B == 0:
            break

        sum_A += k * p[k]
        mean_A = sum_A / weight_A if weight_A != 0 else 0
        mean_B = (mg - sum_A) / weight_B if weight_B != 0 else 0

        between_class_variance = weight_A * weight_B * (mean_A - mean_B) ** 2

        if between_class_variance > max_between_class_variance:
            max_between_class_variance = between_class_variance
            threshold_value = k
            m1 = mean_A
            m2 = mean_B

    return threshold_value, mg, m1, m2

# Hàm tính ngưỡng Global Mean
def global_mean_threshold(image):
    return np.mean(image)

# Hàm tìm ngưỡng đa ngưỡng
def multi_threshold(image):
    histogram, _ = np.histogram(image, bins=256, range=(0, 256))
    total_pixels = image.size
    max_variance = 0
    best_thresholds = (0, 0)

    for t1 in range(1, 255):
        for t2 in range(t1 + 40, 256):
            w1 = np.sum(histogram[:t1])
            w2 = np.sum(histogram[t1:t2])
            w3 = total_pixels - w1 - w2

            if w1 == 0 or w2 == 0 or w3 == 0:
                continue

            m1 = np.sum(np.arange(t1) * histogram[:t1]) / w1
            m2 = np.sum(np.arange(t1, t2) * histogram[t1:t2]) / w2
            m3 = np.sum(np.arange(t2, 256) * histogram[t2:]) / w3

            variance_between = (w1 * (m1 - (w1 + w2 + w3) / total_pixels) ** 2 + 
                                w2 * (m2 - (w1 + w2 + w3) / total_pixels) ** 2 + 
                                w3 * (m3 - (w1 + w2 + w3) / total_pixels) ** 2)

            if variance_between > max_variance:
                max_variance = variance_between
                best_thresholds = (t1, t2)

    return best_thresholds

# Hàm xử lý ảnh và hiển thị kết quả
def process_images_otsu():
    clear_previous_images()
    img_array = cv2.imread(original_image.get(), cv2.IMREAD_GRAYSCALE)
    if img_array is not None:
        thresholds_otsu = otsu_threshold(img_array)[0]
        segmented_otsu = (img_array >= thresholds_otsu).astype(np.uint8) * 255
        display_processed_image(segmented_otsu, otsu_title, "Ảnh đã xử lý bằng Otsu")

def process_images_global_mean():
    clear_previous_images()
    img_array = cv2.imread(original_image.get(), cv2.IMREAD_GRAYSCALE)
    if img_array is not None:
        thresholds_global_mean = global_mean_threshold(img_array)
        segmented_global_mean = (img_array >= thresholds_global_mean).astype(np.uint8) * 255
        display_processed_image(segmented_global_mean, global_mean_title, "Ảnh đã xử lý bằng Global Mean")

def process_images_multi_threshold():
    clear_previous_images()
    img_array = cv2.imread(original_image.get(), cv2.IMREAD_GRAYSCALE)
    if img_array is not None:
        thresholds_multi = multi_threshold(img_array)
        thresholded_img = np.zeros(img_array.shape, dtype=np.uint8)
        t1, t2 = thresholds_multi
        thresholded_img[img_array < t1] = 85
        thresholded_img[(img_array >= t1) & (img_array < t2)] = 170
        thresholded_img[img_array >= t2] = 255
        display_processed_image(thresholded_img, multi_thresholded_title, "Ảnh đã xử lý bằng Đa ngưỡng")

# Hàm xử lý tất cả ảnh và hiển thị kết quả
def process_all_images():
    clear_previous_images()
    img_array = cv2.imread(original_image.get(), cv2.IMREAD_GRAYSCALE)
    if img_array is not None:
        # Xử lý từng thuật toán
        thresholds_otsu = otsu_threshold(img_array)[0]
        segmented_otsu = (img_array >= thresholds_otsu).astype(np.uint8) * 255

        thresholds_global_mean = global_mean_threshold(img_array)
        segmented_global_mean = (img_array >= thresholds_global_mean).astype(np.uint8) * 255

        thresholds_multi = multi_threshold(img_array)
        thresholded_img = np.zeros(img_array.shape, dtype=np.uint8)
        t1, t2 = thresholds_multi
        thresholded_img[img_array < t1] = 85
        thresholded_img[(img_array >= t1) & (img_array < t2)] = 170
        thresholded_img[img_array >= t2] = 255

        # Hiển thị tất cả các ảnh
        display_all_processed_images(segmented_otsu, segmented_global_mean, thresholded_img)

# Hàm hiển thị tất cả ảnh đã xử lý
def display_all_processed_images(otsu_img, global_mean_img, multi_threshold_img):
    # Tạo ảnh cho từng kết quả
    otsu_processed_img = ImageTk.PhotoImage(image=Image.fromarray(otsu_img))
    global_mean_processed_img = ImageTk.PhotoImage(image=Image.fromarray(global_mean_img))
    multi_threshold_processed_img = ImageTk.PhotoImage(image=Image.fromarray(multi_threshold_img))

    # Hiển thị ảnh trên cùng một hàng
    otsu_label.config(image=otsu_processed_img)
    otsu_label.image = otsu_processed_img
    global_mean_label.config(image=global_mean_processed_img)
    global_mean_label.image = global_mean_processed_img
    multi_threshold_label.config(image=multi_threshold_processed_img)
    multi_threshold_label.image = multi_threshold_processed_img

    otsu_title.set("Ảnh đã xử lý bằng Otsu")
    global_mean_title.set("Ảnh đã xử lý bằng Global Mean")
    multi_thresholded_title.set("Ảnh đã xử lý bằng Đa ngưỡng")

# Hàm hiển thị ảnh đã xử lý
def display_processed_image(processed_img, title_var, title_text):
    processed_image = ImageTk.PhotoImage(image=Image.fromarray(processed_img))
    otsu_label.config(image=processed_image)
    otsu_label.image = processed_image
    title_var.set(title_text)

# Hàm xóa ảnh và tiêu đề cũ
def clear_previous_images():
    otsu_label.config(image='')
    global_mean_label.config(image='')
    multi_threshold_label.config(image='')
    otsu_title.set('')
    global_mean_title.set('')
    multi_thresholded_title.set('')

# Tạo giao diện người dùng
root = Tk()
root.title("Nhóm 9 Xử lý ảnh và thị giác máy tính")
root.geometry("1200x800")  # Thay đổi kích thước cửa sổ
root.configure(bg='#e6f7ff')  # Thay đổi màu nền

original_image = StringVar()
otsu_title = StringVar()
global_mean_title = StringVar()
multi_thresholded_title = StringVar()

# Tạo khung cho các nút
button_frame = Frame(root, bg='#e6f7ff')
button_frame.pack(pady=20)

select_button = Button(button_frame, text="Chọn Hình Ảnh", command=select_image, bg='#66b3ff', fg='white', font=("Arial", 14, 'bold'))
select_button.pack(side="left", padx=10)

otsu_button = Button(button_frame, text="Xử lý Otsu", command=process_images_otsu, bg='#ffb3b3', fg='white', font=("Arial", 14, 'bold'))
otsu_button.pack(side="left", padx=10)

global_mean_button = Button(button_frame, text="Xử lý Global Mean", command=process_images_global_mean, bg='#ffb3b3', fg='white', font=("Arial", 14, 'bold'))
global_mean_button.pack(side="left", padx=10)

multi_threshold_button = Button(button_frame, text="Xử lý Đa ngưỡng", command=process_images_multi_threshold, bg='#ffb3b3', fg='white', font=("Arial", 14, 'bold'))
multi_threshold_button.pack(side="left", padx=10)

process_all_button = Button(button_frame, text="Xử lý Tất Cả", command=process_all_images, bg='#ffb3b3', fg='white', font=("Arial", 14, 'bold'))
process_all_button.pack(side="left", padx=10)

# Khung cho ảnh
image_frame = Frame(root, bg='#e6f7ff')
image_frame.pack(pady=20)

original_label = Label(image_frame)
original_label.grid(row=0, column=0, padx=10)

otsu_label = Label(image_frame)
otsu_label.grid(row=1, column=0, padx=10)
otsu_title_label = Label(image_frame, textvariable=otsu_title, bg='#e6f7ff', font=("Arial", 12, 'bold'))
otsu_title_label.grid(row=2, column=0)

global_mean_label = Label(image_frame)
global_mean_label.grid(row=1, column=1, padx=10)
global_mean_title_label = Label(image_frame, textvariable=global_mean_title, bg='#e6f7ff', font=("Arial", 12, 'bold'))
global_mean_title_label.grid(row=2, column=1)

multi_threshold_label = Label(image_frame)
multi_threshold_label.grid(row=1, column=2, padx=10)
multi_thresholded_title_label = Label(image_frame, textvariable=multi_thresholded_title, bg='#e6f7ff', font=("Arial", 12, 'bold'))
multi_thresholded_title_label.grid(row=2, column=2)

root.mainloop()
