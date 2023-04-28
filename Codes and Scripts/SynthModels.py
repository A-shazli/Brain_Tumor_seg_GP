import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
import tensorflow_addons.layers as tfal
from keras.initializers import RandomNormal
from tensorflow.keras.layers import Input,Conv2D,Conv2DTranspose,LeakyReLU,Activation,Concatenate,Add


class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset


class SqueezeAttention():
    def __init__(self):
        self.model = self.squeeze_attention_unet()

    def conv_block(self, x, num_filters):
        x = L.Conv2D(num_filters, 3, padding="same")(x)
        x = tfal.InstanceNormalization(axis=-1)(x)
        x = L.Activation("relu")(x)

        x = L.Conv2D(num_filters, 3, padding="same")(x)
        x = tfal.InstanceNormalization(axis=-1)(x)
        x = L.Activation("relu")(x)

        return x

    def se_block(self, x, num_filters, ratio=8):
        se_shape = (1, 1, num_filters)
        se = L.GlobalAveragePooling2D()(x)
        se = L.Reshape(se_shape)(se)
        se = L.Dense(num_filters // ratio, activation="relu", use_bias=False)(se)
        se = L.Dense(num_filters, activation="sigmoid", use_bias=False)(se)
        se = L.Reshape(se_shape)(se)
        x = L.Multiply()([x, se])
        return x

    def encoder_block(self, x, num_filters):
        x = self.conv_block(x, num_filters)
        x = self.se_block(x, num_filters)
        p = L.MaxPool2D((2, 2))(x)
        return x, p

    def decoder_block(self, x, s, num_filters):
        x = L.UpSampling2D(interpolation="bilinear")(x)
        x = L.Concatenate()([x, s])
        x = self.conv_block(x, num_filters)
        x = self.se_block(x, num_filters)
        return x

    def squeeze_attention_unet(self, input_shape=(256, 256, 3)):
        """ Inputs """
        inputs = L.Input(input_shape)

        """ Encoder """
        s1, p1 = self.encoder_block(inputs, 64)
        s2, p2 = self.encoder_block(p1, 128)
        s3, p3 = self.encoder_block(p2, 256)
        s4, p4 = self.encoder_block(p3, 256)

        b1 = self.conv_block(p4, 512)
        b1 = self.se_block(b1, 512)

        """ Decoder """
        d = self.decoder_block(b1, s4, 256)
        d1 = self.decoder_block(d, s3, 256)
        d2 = self.decoder_block(d1, s2, 128)
        d3 = self.decoder_block(d2, s1, 64)

        """ Outputs """
        outputs = L.Conv2D(3, (1, 1), activation='tanh')(d3)

        """ Model """

        model = Model(inputs, outputs, name="Squeeze-Attention-UNET")
        return model

class ResNet():
    def __init__(self):
        self.model = self.resnet_generator()

    def resnet_block(self, n_filters, input_layer):
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # first layer convolutional layer
        g = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=init)(input_layer)
        g = tfal.InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)
        # second convolutional layer
        g = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=init)(g)
        g = tfal.InstanceNormalization(axis=-1)(g)
        # concatenate merge channel-wise with input layer
        g = Concatenate()([g, input_layer])
        return g

    def resnet_generator(self, image_shape=(256, 256, 3), n_resnet=7):
        # weight initialization
        init = RandomNormal(stddev=0.02)
        in_image = Input(shape=image_shape)
        g = Conv2D(64, (7, 7), padding='same', kernel_initializer=init)(in_image)
        g = tfal.InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)
        g = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
        g = tfal.InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)

        g = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
        g = tfal.InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)

        for _ in range(n_resnet):
            g = self.resnet_block(256, g)

        g = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
        g = tfal.InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)

        g = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
        g = tfal.InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)

        g = Conv2D(3, (7, 7), padding='same', kernel_initializer=init)(g)
        g = tfal.InstanceNormalization(axis=-1)(g)
        out_image = Activation('tanh')(g)
        # define model
        model = Model(in_image, out_image)
        return model


class unet_model():
    def __init__(self):
        self.model = self.unet_generator()

    def downsample(self, filters, size, apply_norm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                          kernel_initializer=initializer, use_bias=False))
        if apply_norm:
            result.add(InstanceNormalization())
        result.add(tf.keras.layers.LeakyReLU())
        return result

    def upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                                   kernel_initializer=initializer, use_bias=False))
        result.add(InstanceNormalization())
        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))
        result.add(tf.keras.layers.ReLU())
        return result

    def unet_generator(self):
        down_stack = [
            self.downsample(64, 4, False),
            self.downsample(128, 4),
            self.downsample(128, 4),
            self.downsample(128, 4),
            self.downsample(128, 4)
        ]
        up_stack = [
            self.upsample(128, 4, True),
            self.upsample(128, 4, True),
            self.upsample(128, 4),
            self.upsample(64, 4)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same', kernel_initializer=initializer,
                                               activation='tanh')
        concat = tf.keras.layers.Concatenate()
        inputs = tf.keras.layers.Input(shape=[256, 256, 3])
        x = inputs
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)
        skips = reversed(skips[:-1])
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = concat([x, skip])
        x = last(x)
        return tf.keras.Model(inputs=inputs, outputs=x)


class Discriminator(unet_model):
    def __init__(self):
        self.model = self.discriminator()

    def discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)
        inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
        x = inp
        down1 = self.downsample(64, 4, False)(x)  # (bs, 16, 16, 64)
        down2 = self.downsample(128, 4)(down1)
        down3 = self.downsample(256, 4)(down2)
        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer,
                                      use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)
        norm1 = InstanceNormalization()(conv)
        leaky_relu = tf.keras.layers.LeakyReLU()(norm1)
        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)
        last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)
        return tf.keras.Model(inputs=inp, outputs=last)

class old_squeeze():
    def __init__(self):
        self.model = self.squeeze_attention_unet()

    def conv_block(self, x, num_filters):
        x = L.Conv2D(num_filters, 3, padding="same")(x)
        x = tfal.InstanceNormalization(axis=-1)(x)
        x = L.Activation("relu")(x)

        x = L.Conv2D(num_filters, 3, padding="same")(x)
        x = tfal.InstanceNormalization(axis=-1)(x)
        x = L.Activation("relu")(x)
        return x

    def se_block(self, x, num_filters, ratio=8):
        se_shape = (1, 1, num_filters)
        se = L.GlobalAveragePooling2D()(x)
        se = L.Reshape(se_shape)(se)
        se = L.Dense(num_filters // ratio, activation="relu", use_bias=False)(se)
        se = L.Dense(num_filters, activation="sigmoid", use_bias=False)(se)
        se = L.Reshape(se_shape)(se)
        x = L.Multiply()([x, se])
        return x

    def encoder_block(self, x, num_filters):
        x = self.conv_block(x, num_filters)
        x = self.se_block(x, num_filters)
        p = L.MaxPool2D((2, 2))(x)
        return x, p

    def decoder_block(self, x, s, num_filters):
        x = L.UpSampling2D(interpolation="bilinear")(x)
        x = L.Concatenate()([x, s])
        x = self.conv_block(x, num_filters)
        x = self.se_block(x, num_filters)
        return x

    def squeeze_attention_unet(self, input_shape=(256, 256, 3)):
        """ Inputs """
        inputs = L.Input(input_shape)

        """ Encoder """
        s1, p1 = self.encoder_block(inputs, 64)
        s2, p2 = self.encoder_block(p1, 128)
        s3, p3 = self.encoder_block(p2, 256)

        b1 = self.conv_block(p3, 512)
        b1 = self.se_block(b1, 512)

        """ Decoder """
        d1 = self.decoder_block(b1, s3, 256)
        d2 = self.decoder_block(d1, s2, 128)
        d3 = self.decoder_block(d2, s1, 64)

        """ Outputs """
        outputs = L.Conv2D(3, (1, 1), activation='tanh')(d3)

        """ Model """
        model = Model(inputs, outputs, name="Squeeze-Attention-UNET")
        return model