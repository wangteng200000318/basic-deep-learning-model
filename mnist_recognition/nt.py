import tensorflow as tf

resnet = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
resnet.summary()
