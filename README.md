# ASLtoText

This project is a simple ASL hand sign recognition app.

I created a script to capture pictures of different hand signs from a webcam. Those images were then uploaded to Google Teachable Machine, labeled by sign, and exported as a Keras model file (`keras_model.h5`) and a label file (`labels.txt`).

The project uses TensorFlow/Keras, OpenCV, and `cvzone` to detect a hand from the webcam feed, preprocess the image, and predict the sign in real time.

## Workflow

1. Capture images of different hand signs.
2. Upload the images to Google Teachable Machine.
3. Label the images by sign.
4. Export the trained `keras_model.h5` and `labels.txt` files.
5. Use TensorFlow to run live predictions from the webcam.

## Files

- `training.py` captures and preprocesses hand images from the webcam.
- `test.py` loads the exported model and performs live sign prediction.
- `Model/labels.txt` contains the label names used by the model.
