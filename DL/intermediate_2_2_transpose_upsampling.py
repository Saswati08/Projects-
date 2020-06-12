from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2DTranspose

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
          b = transpose_upsample(b)
          b = cpy = srcnn(b)
        # if i == 15:
        #   b = upsample(b, 2, num_filters, 'upscale_2')

    b = Conv2D(num_filters, 3, padding='same')(b)
    # print(tf.keras.backend.shape(b))
    # x = Add()([x, b])
    # print(tf.keras.backend.shape(b))
    b = Add()([b, cpy])
    x = transpose_upsample(b)
    x = srcnn(x)
    # x = upsample(x, 4, num_filters, 'upscale_3')
    # print(tf.keras.backend.shape(b))
    x = Conv2D(3, 3, padding='same')(x)
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

def interpolation_upsample(x, do_conv):
  # filters, kernel_size, strides=(1, 1), padding='valid'
  # x = Conv2DTranspose(num_filters, 1, strides = (2, 2), padding ='valid')
  x = UpSampling2D((2, 2), input_shape =tf.shape(x) , interpolation='bilinear')(x)
  # if do_conv:
  #   x = Conv2D(64, 3, padding='same', activation='relu')(x)
  return x

def transpose_upsample(x):
  initializer = tf.random_normal_initializer(0., 0.02)
  x = Conv2DTranspose(64, (1, 1), strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False)(x)
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

def srcnn(x_in):
  x = Conv2D(filters = 128, kernel_size = 9, kernel_initializer='glorot_uniform',
	                     activation='relu', padding='same')(x_in)
  
  x = Conv2D(filters = 64, kernel_size = 3, kernel_initializer='glorot_uniform',
	                     activation='relu', padding='same')(x)
  
  x = Conv2D(filters = 64, kernel_size =5, kernel_initializer='glorot_uniform',
	                     activation='relu', padding='same')(x)
  return x
