import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

class Cnn(object):
    def data_preprocessing(self):
        # Data Preprocessing

        # Preprocessing the Training set
        train_datagen = ImageDataGenerator(rescale = 1./255,
                                           shear_range = 0.2,
                                           zoom_range = 0.2,
                                           horizontal_flip = True)
        training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 shuffle=True,
                                                 class_mode = 'binary')

        # Preprocessing the Test set
        test_datagen = ImageDataGenerator(rescale = 1./255)
        test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
        # Building the CNN

        # Initialising the CNN
        cnn = tf.keras.models.Sequential()

        # Convolution
        cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

        # Pooling
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        cnn.add(tf.keras.layers.Dropout(0.2))

        # Adding a second convolutional layer
        cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        cnn.add(tf.keras.layers.Dropout(0.2))
        
        # Flattening
        cnn.add(tf.keras.layers.Flatten())

        # Full Connection
        cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

        # Output Layer
        cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

        # Compiling the CNN
        cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

        # Training the CNN on the Training set and evaluating it on the Test set
        cnn.fit(x = training_set, validation_data = test_set, epochs = 25)
        cnn.save("my_model.json")

def load_files():
    path = './dataset/prediction/'
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.jpg' in file:
                files.append(os.path.join(r, file))
    return files

def main():
    while True:
        print("1-train")
        print("2-test")
        print("3-info")
        op = int(input())
        if op == 1:
            model = Cnn()
            model.data_preprocessing()
        elif op == 2:
            model = tf.keras.models.load_model("my_model.json")
            images = load_files()
            for files in images:
                test_image = image.load_img(files, target_size = (64, 64))
                test_image = image.img_to_array(test_image)
                test_image = np.expand_dims(test_image, axis = 0)
                result = model.predict(test_image)
                if result[0][0] == 1:
                    prediction = 'dog'
                else:
                    prediction = 'cat'

                original_image = mpimg.imread(files)
                fig, ax = plt.subplots()
                plt.xlabel("prediction: " + prediction)
                ax.imshow(original_image)
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.show()
            
        elif op == 3:
            model = tf.keras.models.load_model("my_model.json")
            model.summary()
        else:
            print("Invalid option\n")
if __name__ == '__main__':
    main()
