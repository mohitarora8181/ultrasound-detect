from flask import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
import urllib.request
from PIL import Image

model = load_model('final_model.h5') 

# Define class labels
class_labels = ['benign', 'malignant', 'normal']


app = Flask(__name__,template_folder="templates")

dataset1 = []

def upload_image(url):
    image = cv2.imread(url)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((224, 224))
    image = np.array(image)
    dataset1.append(image)

# upload_image("/content/check.png")

def classify_image(dataset1):
		predicted_class_label = None
		input_image = np.expand_dims(dataset1[0], axis=0)
		prediction = model.predict(input_image)
		predicted_class_index = np.argmax(prediction)
		predicted_class_label = class_labels[predicted_class_index]
		# plt.imshow(input_image.squeeze())
		# plt.title(f"Predicted Class: {predicted_class_label}")
		# plt.axis('off')
		# plt.show()
		return predicted_class_label


@app.route("/" , methods=['GET','POST'])
def hme():
  return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_out():
	if request.method == 'POST':
		img = request.files['my_image']
		img.save("static/inputs.png")
		dataset1.clear()
		upload_image("static/inputs.png")
		result = classify_image(dataset1)

	return render_template("index.html" ,predict = result, img_path = "inputs.png")
