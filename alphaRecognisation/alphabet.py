import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time

X = np.load("image.npz")["arr_0"]
y = pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())
classes = ['A', 'B', 'C','D', 'E','F', 'G', 'H', 'I', 'J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
nclasses = len(classes)

samplesPerClass  = 5
figure = plt.figure(figsize = (nclasses*2,(1+samplesPerClass*2)))
idx_cls = 0
for cls in classes:
  idxs = np.flatnonzero(y == cls)
  idxs = np.random.choice(idxs,samplesPerClass, replace = False)
  i = 0
  for idx in idxs :
    plt_idx = i * nclasses + idx_cls + 1
    p = plt.subplot(samplesPerClass,nclasses,plt_idx)
    p = sns.heatmap(np.reshape(X[idx],(22,30)),cmap = plt.cm.gray,xticklabels= False,yticklabels= False , cbar = False)
    p = plt.axis("off")
    i = i+1
  
  idx_cls += 1

  xTrain,xTest,yTrain,yTest = train_test_split(X,y,random_state = 9, train_size = 7500,test_size= 2500)
xTrainScaled = xTrain/255.0
xTestScaled = xTest/255.0

clf = LogisticRegression(solver = "saga",multi_class = "multinomial").fit(xTrainScaled,yTrain)
yPrediction = clf.predict(xTestScaled)
accuracy = accuracy_score(yTest,yPrediction)
print(accuracy)



cap = cv2.VideoCapture(0)
while True:
    try:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,width = gray.shape
        upperLeft = (int(width/2 - 56),int(height/2 -56))
        bottomRight = (int(width/2 +56), int(height/2 + 56))
        cv2.rectangle(gray,upperLeft,bottomRight,(0,255,0),2)
        roi = gray[upperLeft[1]: bottomRight[1], upperLeft[0]: bottomRight[0]]
        image_pil = Image.fromarray(roi)
        image_bw = image_pil.convert("L")
        image_bw_resize = image_bw.resize((28,28),Image.ANTIALIAS)
        image_bw_resize_invert = PIL.ImageOps.invert(image_bw_resize)
        pixelFilter = 20
        min_pixel = np.percentile(image_bw_resize_invert,pixelFilter)
        image_bw_resize_invert_scaled = np.clip(image_bw_resize_invert-min_pixel,0,255)
        maxPixel = np.max(image_bw_resize_invert)
        image_bw_resize_invert_scaled = np.asarray(image_bw_resize_invert_scaled)/maxPixel

        testSample = np.array(image_bw_resize_invert_scaled).reshape(1,784)
        testPrediction = clf.predict(testSample)
        print("predicted Class is :", testPrediction)
        cv2.imshow("frame",gray)


    except exception as e:
        pass
cap.release()
cv2.distroyAllWindows()