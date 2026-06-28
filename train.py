from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

train = ImageDataGenerator(rescale=1./255)

train_data = train.flow_from_directory(
    r'D:\DS_PRACTICE\26-06-2026\HappySadCNN\dataset\Train',
    target_size=(200,200),
    batch_size=32,
    class_mode='binary'
)
model = Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(200,200,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(train_data,epochs=10)

print(train_data.class_indices)

model.save("emotion_model.h5")