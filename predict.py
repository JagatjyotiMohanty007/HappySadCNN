import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("model.h5")
img = image.load_img(
    r'D:\DS_PRACTICE\26-06-2026\HappySadCNN\dataset\Train\Happpy\JJ.jpeg',
    target_size=(150,150)
)
# img=image.load_img("test.jpeg",target_size=(150,150))
img=image.img_to_array(img)
img=np.expand_dims(img,axis=0)
img=img/255.0

result=model.predict(img)


if result[0][0]>0.5:
    print("Sad")
else:
    print("Happy")