from django.shortcuts import render
import tensorflow as tf
import cv2
import numpy as np
import requests

ipfs_url = "https://ipfs.io/ipfs/bafybeicyix56ixaapdfh3ob62h526m4sig2raqga72apub4lpdzouvlryi"

# Make the request to download the file
response = requests.get(ipfs_url)

model_response = requests.get(ipfs_url)

with open("cancer_model_downloaded.h5", "wb") as f:
    f.write(model_response.content)
model = tf.keras.models.load_model('cancer_model_downloaded.h5')

def home(request):
    message = None
    if request.method == 'POST' and request.FILES.get('image'):
        # Get the uploaded image from the request object
        image = request.FILES['image'].read()

        # Decode the image and resize it to 50x50 pixels
        npimg = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (50, 50))

        # Preprocess the image and make a prediction
        img = img.reshape((1, 50, 50, 1)) / 255.0 
        prediction = model.predict(img)

        # Set the message based on the prediction
        if prediction.argmax() == 0:
            message = "Lung Cancer Detected"
        else:
            message = "Lung Cancer Not Detected"

    return render(request, 'home.html', {'message': message})
