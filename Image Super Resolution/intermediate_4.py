from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda
from tensorflow.python.keras.models import Model

# from model.common import normalize, denormalize, pixel_shuffle


def edsr(scale, num_filters=64, num_res_blocks=8, res_block_scaling=None):
    x_in = Input(shape=(None, None, 3))
    
    # print(tf.keras.backend.shape(x_in))
    x = Lambda(normalize)(x_in)
    # print(tf.keras.backend.shape(x))
    x = b = Conv2D(num_filters, 3, padding='same')(x)
    # print(tf.keras.backend.shape(x))
    for i in range(num_res_blocks):
        
        b = res_block(b, num_filters, res_block_scaling)
        print(tf.keras.backend.shape(b))
        if i == 8:
          b = Add()([x, b])
          b = cpy = upsample(b, 4, num_filters, 'upscale_1')
        # if i == 15:
        #   b = upsample(b, 2, num_filters, 'upscale_2')

    b = Conv2D(num_filters, 3, padding='same')(b)
    # print(tf.keras.backend.shape(b))
    # x = Add()([x, b])
    # print(tf.keras.backend.shape(b))
    b = Add()([b, cpy])
    # x = upsample(b, 2, num_filters, 'upscale_2')
    # x = upsample(x, 4, num_filters, 'upscale_3')
    # print(tf.keras.backend.shape(b))
    x = Conv2D(3, 3, padding='same')(b)
    # print(tf.keras.backend.shape(b))

    x = Lambda(denormalize)(x)
    # print(tf.keras.backend.shape(b))
    # print("Done")
    return Model(x_in, x, name="edsr")


def res_block(x_in, filters, scaling):
    x = Conv2D(filters, 3, padding='same', activation='relu')(x_in)
    x = Conv2D(filters, 3, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def upsample(x, scale, num_filters, name):
    def upsample_1(x, factor, **kwargs):
        x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        return Lambda(pixel_shuffle(scale=factor))(x)

    if scale == 2:
        x = upsample_1(x, 2, name= name)
    elif scale == 3:
        x = upsample_1(x, 3, name=name)
    elif scale == 4:
        x = upsample_1(x, 2, name = name+'1')
        x = upsample_1(x, 2, name= name+ '2')

    return x
