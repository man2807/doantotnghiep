import cv2
import os
import numpy as np
import face_recognition
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

def draw_ped(img, label, x0, y0, xt, yt, color=(255,127,0), text_color=(255,255,255)):

    (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img,
                  (x0, y0 + baseline),  
                  (max(xt, x0 + w), yt), 
                  color, 
                  2)
    cv2.rectangle(img,
                  (x0, y0 - h),  
                  (x0 + w, y0 + baseline), 
                  color, 
                  -1)  
    cv2.putText(img, 
                label, 
                (x0, y0),                   
                cv2.FONT_HERSHEY_SIMPLEX,     
                0.5,                          
                text_color,                
                1,
                cv2.LINE_AA) 
    return img

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
    imgs.append(cv2.warpAffine(img, M_rot_5, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(img, M_rot_neg_5, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(img, M_rot_10, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(img, M_rot_neg_10, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(img, M_trans_3, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(img, M_trans_neg_3, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(img, M_trans_6, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(img, M_trans_neg_6, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(img, M_trans_y3, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(img, M_trans_neg_y3, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(img, M_trans_y6, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(img, M_trans_neg_y6, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.add(img, 10))
    imgs.append(cv2.add(img, 30))
    imgs.append(cv2.add(img, -10))
    imgs.append(cv2.add(img, -30)) 
    imgs.append(cv2.add(img, 15))
    imgs.append(cv2.add(img, 45))
    imgs.append(cv2.add(img, -15))
    imgs.append(cv2.add(img, -45))
    
    return imgs

def print_data(label_distr, label_name):
    plt.figure(figsize=(12,6))

    my_circle = plt.Circle( (0,0), 0.7, color='white')
    plt.pie(label_distr, labels=label_name, autopct='%1.1f%%')
    plt.gcf().gca().add_artist(my_circle)
    plt.show()


def randc(labels, l):
    return np.random.choice(np.where(np.array(labels) == l)[0], 100, replace=1)

def detect_face(img):
    # img = img[70:195,78:172]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (50, 50), interpolation =cv2.INTER_AREA)
    # img = cv2.resize(img, (151, 151))
    return img

def print_progress(val, val_len, folder, bar_size=20):
    progr = "#"*round((val)*bar_size/val_len) + " "*round((val_len - (val))*bar_size/val_len)
    if val == 0:
        print("", end = "\n")
    else:
        print("[%s] (%d samples)\t label : %s \t\t" % (progr, val+1, folder), end="\r")

def main():
    dataset_folder = "D:/Face_recognition_yen/Face_recognition_2/train/"

    names = []
    images = []
    for folder in os.listdir(dataset_folder):
        files = os.listdir(os.path.join(dataset_folder, folder))[:150]
        # if len(files) < 50 :
        #     continue
        for i, name in enumerate(files): 
            if name.find(".jpg") > -1 :
                img = cv2.imread(os.path.join(dataset_folder + folder, name))
                img = detect_face(img) # detect face using mtcnn and crop to 100x100
                if img is not None :
                    images.append(img)
                    names.append(folder)
                    print_progress(i, len(files), folder)

    print("number of samples :", len(names))

    img_test = images[0]

    augmented_image_test = img_augmentation(img_test)

    plt.figure(figsize=(15,10))
    for i, img in enumerate(augmented_image_test):
        plt.subplot(4,5,i+1)
        plt.imshow(img, cmap="gray")
    #plt.show()

    augmented_images = []
    augmented_names = []
    for i, img in enumerate(images):
        try :
            augmented_images.extend(img_augmentation(img))
            augmented_names.extend([names[i]] * 20)
        except :
            print(i)

    len(augmented_images), len(augmented_names)


    images.extend(augmented_images)
    names.extend(augmented_names)

    len(images), len(names)

    unique, counts = np.unique(names, return_counts = True)

    for item in zip(unique, counts):
        print(item)

    unique = np.unique(names)
    label_distr = {i:names.count(i) for i in names}.values()
    #print_data(label_distr, unique)

    mask = np.hstack([randc(names, l) for l in np.unique(names)])
    names = [names[m] for m in mask]
    images = [images[m] for m in mask]
    label_distr = {i:names.count(i) for i in names}.values()
    #print_data(label_distr, unique)

    le = LabelEncoder()
    le.fit(names)
    labels = le.classes_
    name_vec = le.transform(names)
    categorical_name_vec = to_categorical(name_vec)
    print("number of class :", len(labels))
    print(labels)

    # --------- load Haar Cascade model -------------
    face_cascade = cv2.CascadeClassifier('D:/Face_recognition_yen/Face_recognition_2/haarcascade_frontalface_default.xml')

    # --------- load Keras CNN model -------------
    model = load_model("D:/Face_recognition_yen/Face_recognition_2/model-cnn-facerecognition.h5")
    print("[INFO] finish load model...")

    cap = cv2.VideoCapture(0)
    while cap.isOpened() :
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            boxes = face_recognition.face_locations(gray,model="hog")
            #faces = face_cascade.detectMultiScale(gray, 1.4, 3)
            for (x, y, w, h) in boxes:
                
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (50, 50),interpolation =cv2.INTER_AREA)
                face_img = face_img.reshape(1, 50, 50, 1)
                face_img = tf.cast(face_img, tf.float32)
                
                result = model.predict(face_img)
                idx = result.argmax(axis=1)
                confidence = result.max(axis=1)*100
                if confidence > 80:
                    label_text = "%s (%.2f %%)" % (labels[idx], confidence)
                else :
                    label_text = "N/A"
                frame = draw_ped(frame, label_text, x, y, x + w, y + h, color=(0,255,255), text_color=(50,50,50))
        
            cv2.imshow('Detect Face', frame)
        else :
            break
        if cv2.waitKey(10) == ord('q'):
            break
            
    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()