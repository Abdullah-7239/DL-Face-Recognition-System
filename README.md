# Deep-Learning-Face-Recognition

## Step-1 Run faceseamlessentry.ipynb 
To generate the images of the people using the webcam. You can adjust the number of images to your suiting. My preference for better Deep Learning Model Classification use more than 100 images per person in each set(train,test).

## Step-2 Organize the File format 
While storing the images per person your folder structure shall be as such :
/Dataset/Train/Person1/image.jpg
/Dataset/Test/Person1/image.jpg

## Step-3 Run facerecognition.py
This will train the model on the images you just took from ipynb file. You can customuze the Deep Learning Model to your suiting.

## Step-4 Run facefrontend.py
This will open your camera and keep predicting the person based on the faces it detects. You can stop it by hitting "Q" key.
