# example of progressively loading images from file
# D:/Documents/Python/CH27/Minh/data/train

from keras.layers import Conv2D, MaxPool2D, Flatten

import os
import cv2
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input

from keras.utils import to_categorical
from numpy import *

def detect_face(img):
    img = img[70:195,78:172]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (50, 50))
    return img

'''
#def detect_face(img):
      
      detect = cv2.CascadeClassifier('D:/Documents/Python/CH27/Minh/haarcascade_frontalface_default.xml')
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      #faces = detect.detectMultiScale(gray)   
      faces = detect.detectMultiScale(gray,
                                 scaleFactor=1.3, 
                                 minNeighbors=4, 
                                 minSize=(50, 50),
                                 flags=cv2.CASCADE_SCALE_IMAGE)
      for (x, y, w, h) in faces:
            print(x, y, w, h)
            faces = gray[y:y + h, x:x + w]
            cv2.imshow("hello", faces)
            cv2.waitKey()
      
      img = cv2.resize(faces, (50, 50), interpolation = cv2.INTER_AREA)
      return img
'''

def print_progress(val, val_len, folder, bar_size=20):
    progr = "#"*round((val)*bar_size/val_len) + " " * \
        round((val_len - (val))*bar_size/val_len)
    if val == 0:
        print("", end="\n")
    else:
        print("[%s] (%d samples)\t label : %s \t\t" %
              (progr, val+1, folder), end="\r")


#dataset_folder = "D:/Documents/Python/CH27/Minh/data/train/"

#dataset_folder = "C:/Users/Administrator/Downloads/lfw-deepfunneled/lfw-deepfunneled/"

dataset_folder = "F:/DEEP_LEARNING/Face_recognition_2/Anhmau/"


names = []
images = []
for folder in os.listdir(dataset_folder):
    files = os.listdir(os.path.join(dataset_folder, folder))[:150]
    #if len(files) < 50:
    #      continue
    for i, name in enumerate(files):
          if name.find(".jpg") > -1:
                img = cv2.imread(os.path.join(dataset_folder + folder, name))
                # detect face using mtcnn and crop to 100x100
                #print(name)
                #cv2.waitKey
                img = detect_face(img)
                
          if img is not None:
                images.append(img)
                names.append(folder)

                print_progress(i, len(files), folder)


print("\nnumber of samples :", len(names))


def img_augmentation(img):
    h, w = img.shape
    center = (w // 2, h // 2)
    M_rot_5 = cv2.getRotationMatrix2D(center, 5, 1.0)
    M_rot_neg_5 = cv2.getRotationMatrix2D(center, -5, 1.0)
    M_rot_10 = cv2.getRotationMatrix2D(center, 10, 1.0)
    M_rot_neg_10 = cv2.getRotationMatrix2D(center, -10, 1.0)
    M_trans_3 = np.float32([[1, 0, 3], [0, 1, 0]])
    M_trans_neg_3 = np.float32([[1, 0, -3], [0, 1, 0]])
    M_trans_6 = np.float32([[1, 0, 6], [0, 1, 0]])
    M_trans_neg_6 = np.float32([[1, 0, -6], [0, 1, 0]])
    M_trans_y3 = np.float32([[1, 0, 0], [0, 1, 3]])
    M_trans_neg_y3 = np.float32([[1, 0, 0], [0, 1, -3]])
    M_trans_y6 = np.float32([[1, 0, 0], [0, 1, 6]])
    M_trans_neg_y6 = np.float32([[1, 0, 0], [0, 1, -6]])

    imgs = []
    imgs.append(cv2.warpAffine(img, M_rot_5, (w, h),
                borderValue=(255, 255, 255)))
    imgs.append(cv2.warpAffine(img, M_rot_neg_5,
                (w, h), borderValue=(255, 255, 255)))
    imgs.append(cv2.warpAffine(img, M_rot_10,
                (w, h), borderValue=(255, 255, 255)))
    imgs.append(cv2.warpAffine(img, M_rot_neg_10,
                (w, h), borderValue=(255, 255, 255)))
    imgs.append(cv2.warpAffine(img, M_trans_3,
                (w, h), borderValue=(255, 255, 255)))
    imgs.append(cv2.warpAffine(img, M_trans_neg_3,
                (w, h), borderValue=(255, 255, 255)))
    imgs.append(cv2.warpAffine(img, M_trans_6,
                (w, h), borderValue=(255, 255, 255)))
    imgs.append(cv2.warpAffine(img, M_trans_neg_6,
                (w, h), borderValue=(255, 255, 255)))
    imgs.append(cv2.warpAffine(img, M_trans_y3,
                (w, h), borderValue=(255, 255, 255)))
    imgs.append(cv2.warpAffine(img, M_trans_neg_y3,
                (w, h), borderValue=(255, 255, 255)))
    imgs.append(cv2.warpAffine(img, M_trans_y6,
                (w, h), borderValue=(255, 255, 255)))
    imgs.append(cv2.warpAffine(img, M_trans_neg_y6,
                (w, h), borderValue=(255, 255, 255)))
    imgs.append(cv2.add(img, 10))
    imgs.append(cv2.add(img, 30))
    imgs.append(cv2.add(img, -10))
    imgs.append(cv2.add(img, -30))
    imgs.append(cv2.add(img, 15))
    imgs.append(cv2.add(img, 45))
    imgs.append(cv2.add(img, -15))
    imgs.append(cv2.add(img, -45))

    return imgs


#plt.imshow(images[1], cmap="gray")

img_test = images[3]

augmented_image_test = img_augmentation(img_test)

plt.figure(figsize=(15,10))
for i, img in enumerate(augmented_image_test):
    plt.subplot(4,5,i+1)
    plt.imshow(img, cmap="gray")
plt.show()

augmented_images = []
augmented_names = []
for i, img in enumerate(images):
    try:
        augmented_images.extend(img_augmentation(img))
        augmented_names.extend([names[i]] * 20)
    except:
        print(i)

len(augmented_images), len(augmented_names)

images.extend(augmented_images)
names.extend(augmented_names)

len(images), len(names)

unique, counts = np.unique(names, return_counts = True)

for item in zip(unique, counts):
    print(item)

# preview data distribution


def print_data(label_distr, label_name):
    plt.figure(figsize=(12, 6))

    my_circle = plt.Circle((0, 0), 0.7, color='white')
    plt.pie(label_distr, labels=label_name, autopct='%1.1f%%')
    plt.gcf().gca().add_artist(my_circle)
    plt.show()


unique = np.unique(names)
label_distr = {i: names.count(i) for i in names}.values()
print_data(label_distr, unique)


# reduce sample size per-class using numpy random choice
n = 100

def randc(labels, l):
    return np.random.choice(np.where(np.array(labels) == l)[0], n, replace=1)

mask = np.hstack([randc(names, l) for l in np.unique(names)])

names = [names[m] for m in mask]
images = [images[m] for m in mask]

label_distr = {i:names.count(i) for i in names}.values()
print_data(label_distr, unique)

len(names)



le = LabelEncoder()

le.fit(names)

labels = le.classes_

name_vec = le.transform(names)

categorical_name_vec = to_categorical(name_vec)

print("number of class :", len(labels))

print(labels)


print(name_vec)

print(categorical_name_vec)

x_train, x_test, y_train, y_test = train_test_split(np.array(images, dtype=np.float32),   # input data
                                                    # target/output data
                                                    np.array(
                                                        categorical_name_vec),
                                                    test_size=0.15,
                                                    random_state=42)

print(x_train.shape, y_train.shape, x_test.shape,  y_test.shape)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)


x_train.shape, x_test.shape


def cnn_model(input_shape):
    model = Sequential()

    model.add(Conv2D(64,
                     (3, 3),
                     padding="valid",
                     activation="relu",
                     input_shape=input_shape))
    model.add(Conv2D(64,
                     (3, 3),
                     padding="valid",
                     activation="relu",
                     input_shape=input_shape))

    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(128,
                     (3, 3),
                     padding="valid",
                     activation="relu"))
    model.add(Conv2D(128,
                     (3, 3),
                     padding="valid",
                     activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(len(labels)))  # equal to number of classes
    model.add(Activation("softmax"))

    model.summary()

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


input_shape = x_train[0].shape

EPOCHS = 10
BATCH_SIZE = 32

model = cnn_model(input_shape)

history = model.fit(x_train,
                    y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    validation_split = 0.15   # 15% of train dataset will be used as validation set
                    )

def evaluate_model_(history):
    names = [['acc', 'val_acc'], 
             ['loss', 'val_loss']]
    for name in names :
        fig1, ax_acc = plt.subplots()
        plt.plot(history.history[name[0]])
        plt.plot(history.history[name[1]])
        plt.xlabel('Epoch')
        plt.ylabel(name[0])
        plt.title('Model - ' + name[0])
        plt.legend(['Training', 'Validation'], loc='lower right')
        plt.grid()
        plt.show()
        
evaluate_model_(history)

model.save("F:/DEEP_LEARNING/Face_recognition_2/model-cnn-facerecognition1.h5")

# predict test data
y_pred = model.predict(x_test)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 8))
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=labels,normalize=False,
                      title='Confusion matrix')

print(classification_report(y_test.argmax(axis=1),
                            y_pred.argmax(axis=1),
                            target_names=labels))


