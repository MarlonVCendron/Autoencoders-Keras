# -*- coding: utf-8 -*-
import numpy as np
import matplotlib as mpl
mpl.use("Qt4Agg")
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras import regularizers

(x_train, _), (x_test, _) = mnist.load_data()

# Normaliza os vlaores para ficarem entre 0 e 1
max_value = float(x_train.max())
x_train = x_train.astype('float32') / max_value
x_test = x_test.astype('float32') / max_value

# Muda a forma de (60000, 28, 28) para (60000, 784)
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# Muda a forma de (10000, 28, 28) para (10000, 784)
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


# ENCODER

# 784
input_dim = x_train.shape[1]
encoding_dim = 32

# 24.5
compression_factor = float(input_dim) / encoding_dim
print("Compression factor: %s" % compression_factor)

autoencoder = Sequential([
    Dense(encoding_dim, input_shape=(input_dim,), activation='relu'),
    Dense(input_dim, activation='sigmoid')
])

# Pega o modelo do encpder para poder ver a imagem codificada
input_img = Input(shape=(input_dim,))
encoder_layer = autoencoder.layers[0]
encoder = Model(input_img, encoder_layer(input_img))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# x_train é input e output, porque é desejado que o output seja a recontrucao mais próxima do input
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

num_images = 10
np.random.seed(42)
random_test_images = np.random.randint(x_test.shape[0], size=num_images)

encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)

plt.figure(figsize=(18, 4))

for i, image_idx in enumerate(random_test_images):
    # plot original image
    ax = plt.subplot(3, num_images, i + 1)
    plt.imshow(x_test[image_idx].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # plot encoded image
    ax = plt.subplot(3, num_images, num_images + i + 1)
    plt.imshow(encoded_imgs[image_idx].reshape(8, 4))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # plot reconstructed image
    ax = plt.subplot(3, num_images, 2 * num_images + i + 1)
    plt.imshow(decoded_imgs[image_idx].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
