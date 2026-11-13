import os
import json
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization, GlobalAveragePooling2D   
from keras.callbacks import EarlyStopping, ReduceLROnPlateau          
from keras.utils import to_categorical
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# define image size
img_rows = 128      # 높이
img_cols = 128      # 너비   

# fix random seed for reproducibility
seed = 123
np.random.seed(seed)
keras.utils.set_random_seed(seed)     # [추가] tensorflow 시드까지 고정 → 실험 결과 재현 가능

# Set data folder
img_dir = './dataset'          # dew, fogsmog, frost ... snow 폴더가 들어있는 경로

# get class names (폴더 이름이 레이블)
class_names = sorted(os.listdir(img_dir))
num_classes = len(class_names)          # 11
print(class_names)

# get file names
flist = []
labels = []
for label, cname in enumerate(class_names):
    for fname in os.listdir(os.path.join(img_dir, cname)):
        flist.append(os.path.join(img_dir, cname, fname))
        labels.append(label)

# load images
X = np.zeros(shape=(len(flist), img_rows, img_cols, 3), dtype=np.float32)
y = np.zeros(shape=(len(flist)), dtype=np.int32)
ok = np.ones(len(flist), dtype=bool)

for idx, img_path in enumerate(flist):
    try:
        img = image.load_img(img_path, target_size=(img_rows, img_cols))  # RGB 로 읽음
        X[idx] = image.img_to_array(img)
        y[idx] = labels[idx]
    except Exception:
        ok[idx] = False                 # 깨진 이미지 파일은 제외

X = X[ok]
y = y[ok]
print('number of images:', len(X))

# scaling into [0, 1]
X /= 255.0          # 제자리 연산 (메모리 절약)

# split train / test  (test 20%, random_state 123)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=123,
                                                    stratify=y)
del X               # 원본 배열 메모리 해제

# train 의 15% 를 validation 으로 분리 (test 는 최종 평가에만 사용)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train,
                                            test_size=0.15,
                                            random_state=123,
                                            stratify=y_train)

# one hot encoding
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
y_tr = to_categorical(y_tr, num_classes)
y_val = to_categorical(y_val, num_classes)

# data augmentation (train 데이터에만 적용)
datagen = ImageDataGenerator(
    horizontal_flip=True,       # randomly flipping
    zoom_range=0.1,             # randomly zooming
    fill_mode='nearest'         # filling in newly created pixels
    )

datagen.fit(X_tr)

train_gen = datagen.flow(X_tr, y_tr, batch_size=32, seed=seed)


# create CNN model
def cnn_model():
    # define model
    model = Sequential()
    model.add(Input(shape=(img_rows, img_cols, 3)))      # 컬러 이미지 → channel 3

    # Conv 블록 4개 : (Conv → BN) x 2 → MaxPooling → Dropout
    for filters in [32, 64, 128, 256]:
        model.add(Conv2D(filters, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())                                  # [추가]
        model.add(Conv2D(filters, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())                                 
        model.add(MaxPooling2D(pool_size=(2, 2)))      # 128 → 64 → 32 → 16 → 8
        model.add(Dropout(0.2))

    model.add(GlobalAveragePooling2D())                
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


# build the model
model = cnn_model()
model.summary()
3
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# Fit the model (augmentation 된 train_gen 으로 학습, batch_size 는 flow 에서 지정)
disp = model.fit(train_gen,
                 validation_data=(X_val, y_val),
                 epochs=80,
                 callbacks=[early_stop, reduce_lr],
                 verbose=1)

# Final evaluation of the model
train_scores = model.evaluate(X_train, y_train, verbose=0)
test_scores = model.evaluate(X_test, y_test, verbose=0)
print("train loss: %.4f   train acc: %.4f" % (train_scores[0], train_scores[1]))
print("test  loss: %.4f   test  acc: %.4f" % (test_scores[0], test_scores[1]))

# summarize history for accuracy
plt.plot(disp.history['accuracy'])
plt.plot(disp.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('accuracy_curve.png')
plt.show()

# save model (Part 2 앱에서 로딩하여 사용)
model.save('weather_cnn.keras')
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)
print('model saved')