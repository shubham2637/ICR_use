import tensorflow as tf
import cv2


tf.get_logger().setLevel('INFO')
print("Version: ", tf.version)
print("Eager mode: ", tf.executing_eagerly())
#print("Hub Version: ", hub.version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")


labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# loading the ICR model
new_model = tf.keras.models.load_model('final_ICR_35_98')

new_model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])


def predictions(image_location):
    img = cv2.imread(image_location)
    grayimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resizeimage = cv2.resize(grayimage, (28, 28))
    imge = resizeimage.reshape(1, 28, 28, 1)
    classes = new_model.predict_classes(imge)
    # print(labels[int(classes)])
    return labels[int(classes)]
