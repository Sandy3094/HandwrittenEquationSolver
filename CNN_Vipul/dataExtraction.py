import os
import cv2
import numpy as np
import random
import pandas as pd

dataDir = "data/extracted_images"

supportedItems = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "-",
    "+",
    "times",
    "div"
]
for folder in supportedItems:
    print(folder + ":" + str(len(os.listdir(dataDir + "/" + folder))))

def pick_random_files(directory, n):
    files = os.listdir(directory)
    random.shuffle(files)
    selected_files = files[:n]
    return selected_files


def load_images_from_folder(folder):
    train_data=[]
    files = pick_random_files(folder, 4000)
    for filename in files:
        img = cv2.imread(os.path.join(folder,filename),cv2.IMREAD_GRAYSCALE)
        img=~img
        if img is not None:
            ret,thresh=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
            ctrs,ret=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            cnt=sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
            w=int(28)
            h=int(28)
            maxi=0
            for c in cnt:
                x,y,w,h=cv2.boundingRect(c)
                maxi=max(w*h,maxi)
                if maxi==w*h:
                    x_max=x
                    y_max=y
                    w_max=w
                    h_max=h
            im_crop= thresh[y_max:y_max+h_max+10, x_max:x_max+w_max+10]
            im_resize = cv2.resize(im_crop,(28,28))
            im_resize=np.reshape(im_resize,(784,1))
            train_data.append(im_resize)
    return train_data

data=None
for index in range(len(supportedItems)):
    folder = supportedItems[index]
    dataImg=load_images_from_folder(dataDir + "/" + folder)
    for i in range(0,len(dataImg)):
        dataImg[i]=np.append(dataImg[i],[index])
    if data is None:
        data=dataImg
    else:
        data=np.concatenate((data,dataImg))

df = pd.DataFrame(data)
df.to_csv('data/processedData.csv', index=False)
