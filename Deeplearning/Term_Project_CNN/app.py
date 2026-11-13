import os
import json
import numpy as np
import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
from PIL import Image, ImageTk

from keras.models import load_model
from keras.preprocessing import image

# define image size (학습할 때와 동일해야 함)
img_rows = 128
img_cols = 128

# 모델 파일은 이 app.py 와 같은 폴더에 있다고 가정 (VS Code 실행 위치와 무관하게 동작)
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'weather_cnn.keras')
class_path = os.path.join(base_dir, 'class_names.json')

# load saved model & class names
model = load_model(model_path)
with open(class_path) as f:
    class_names = json.load(f)


# Load image 버튼을 눌렀을 때 실행
def load_image():
    fname = askopenfilename(title='Select a weather image',
                            filetypes=[('Image files', '*.jpg *.jpeg *.png *.bmp *.gif *.webp'),
                                       ('All files', '*.*')])    # choose image file
    if not fname:
        return

    try:
        img_show = Image.open(fname).convert('RGB')
    except Exception:
        messagebox.showerror('Error', '이미지 파일을 열 수 없습니다.')
        return

    # 화면에 이미지 표시 (비율 유지하며 축소)
    img_show.thumbnail((360, 260))
    photo = ImageTk.PhotoImage(img_show)
    img_label.config(image=photo, text='')
    img_label.image = photo
    file_label.config(text=os.path.basename(fname))

    # 학습할 때와 같은 방법으로 전처리
    img = image.load_img(fname, target_size=(img_rows, img_cols))   # RGB, 128x128
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)          # (1, 128, 128, 3)
    x = x / 255.0                          # scaling into [0, 1]

    # predict
    pred = model.predict(x, verbose=0)[0]
    top3 = np.argsort(pred)[::-1][:3]      # 확률이 높은 순서로 3개

    for i, idx in enumerate(top3):
        result_labels[i].config(text='%d. %-10s (%.2f)' % (i + 1, class_names[idx], pred[idx]))


# ---------------- UI ----------------
window = tk.Tk()
window.title('AI Image Classifier')
window.geometry('640x560')
window.config(bg='white')
window.resizable(False, False)

# 상단 타이틀
tk.Label(window, text='AI Image Classifier', font=('Arial', 15), anchor='w',
         bg='#e8f4fc', padx=12, pady=6).pack(fill='x')

# Load image 버튼
tk.Button(window, text='Load image', font=('Arial', 12), relief='raised', bd=2,
          command=load_image).place(x=30, y=60)

# 이미지 표시 영역
img_label = tk.Label(window, text='No image loaded', fg='gray', bg='#f2f2f2',
                     font=('Arial', 11))
img_label.place(x=210, y=60, width=360, height=260)

file_label = tk.Label(window, text='', fg='gray', bg='white', font=('Arial', 9))
file_label.place(x=210, y=325)

# 결과 표시 영역
tk.Label(window, text='Classification result', font=('Arial', 13),
         bg='white').place(x=210, y=355)
tk.Label(window, text='-' * 50, bg='white').place(x=210, y=380)

result_labels = []
for i in range(3):
    lbl = tk.Label(window, text='', font=('Consolas', 13, 'bold' if i == 0 else 'normal'),
                   fg='#0b5394' if i == 0 else 'black', bg='white', anchor='w')
    lbl.place(x=210, y=405 + i * 32)
    result_labels.append(lbl)

tk.Label(window, text='Top 3', fg='#3c78d8', bg='white',
         font=('Arial', 10)).place(x=500, y=437)

window.mainloop()