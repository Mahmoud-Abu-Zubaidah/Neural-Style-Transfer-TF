# Import libraries
import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
import shutil
import tqdm

Width, hight, channel = 600, 600, 3
tf.random.set_seed(42)

# Style Weight
STYLE_LAYERS = [
    ('block1_conv1', 0.2),
    ('block2_conv1', 0.2),
    ('block3_conv1', 0.2),
    ('block4_conv1', 0.2),
    ('block5_conv1', 0.2)]

# Content Weight
content_layer = [('block5_conv4', 0.025)]


optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

vgg = tf.keras.applications.VGG19(include_top=False,input_shape=(Width,hight, channel))
vgg.trainable = False




def compute_content_cost(content_output, generated_output):
    """
    Computes the content cost

    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

    Returns:
    J_content -- Computes the content cost
    """
    a_C = content_output[-1]
    a_G = generated_output[-1]

    # Retrieve dimensions from a_G (≈1 line)
    _, n_H, n_W, n_C = a_C.shape

    # Reshape 'a_C' and 'a_G' (≈2 lines)
    # DO NOT reshape 'content_output' or 'generated_output'
    a_C_unrolled = tf.transpose(tf.reshape(a_C, shape=[_, -1, n_C]))
    a_G_unrolled = tf.transpose(tf.reshape(a_G, shape=[_, -1, n_C]))

    # compute the cost with tensorflow
    J_content = (1 / (4 * n_H * n_W * n_C)) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))

    return J_content

def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)

    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    GA = tf.matmul(A, A, transpose_b=True)

    return GA

def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

    Returns:
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape the images from (n_H * n_W, n_C) to have them of shape (n_C, n_H * n_W) (≈2 lines)
    a_S = tf.transpose(tf.reshape(a_S, shape=[-1, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, shape=[-1, n_C]))

    # Computing gram_matrices for both images S and G (≈2 lines)
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss (≈1 line)
    J_style_layer = (1 / (4 * n_C **2 * (n_H * n_W) **2)) * tf.reduce_sum(tf.square(tf.subtract(GS, GG)))

    return J_style_layer




def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS=STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers

    Arguments:
    style_image_output -- our tensorflow model
    generated_image_output --
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them

    Returns:
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # Overall style cost
    J_style = 0

    # Set a_S to be the hidden layer activation from the layer we have selected.
    # The last element of the array contains the content layer image, which must not be used.
    a_S = style_image_output[:-1]

    # Set a_G to be the output of the choosen hidden layers.
    # The last element of the list contains the content layer image which must not be used.
    a_G = generated_image_output[:-1]
    for i, weight in zip(range(len(a_S)), STYLE_LAYERS):
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])

        # Upgrade overall style cost
        J_style += weight[1] * J_style_layer

    return J_style


@tf.function()
def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function

    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost

    Returns:
    J -- total cost as defined by the formula above.
    """

    J = alpha * J_content + beta * J_style


    return J


def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS=STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers

    Arguments:
    style_image_output -- our tensorflow model
    generated_image_output --
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them

    Returns:
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # initialize the overall style cost
    J_style = 0

    # Set a_S to be the hidden layer activation from the layer we have selected.
    # The last element of the array contains the content layer image, which must not be used.
    a_S = style_image_output[:-1]

    # Set a_G to be the output of the choosen hidden layers.
    # The last element of the list contains the content layer image which must not be used.
    a_G = generated_image_output[:-1]
    for i, weight in zip(range(len(a_S)), STYLE_LAYERS):
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])

        # Add weight * J_style_layer of this layer to overall style cost
        J_style += weight[1] * J_style_layer

    return J_style

@tf.function()
def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function

    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost

    Returns:
    J -- total cost as defined by the formula above.
    """
    J = alpha * J_content + beta * J_style

    return J



def clip_0_1(image):
    """
    Truncate all the pixels in the tensor to be between 0 and 1

    Arguments:
    image -- Tensor
    J_style -- style cost coded above

    Returns:
    Tensor
    """
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def tensor_to_image(tensor):
    """
    Converts the given tensor into a PIL image

    Arguments:
    tensor -- Tensor

    Returns:
    Image: A PIL image
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

def get_layer_outputs(vgg, layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

def define_ContentAndStyleAndGenerateImage(content_path,syle_path):

    content_image = np.array(Image.open(content_path).resize((Width,hight)))
    imshow(content_image)
    content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))

    style_image =  np.array(Image.open(syle_path).resize((Width,hight)))
    imshow(style_image)
    style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))

    # Create a trainable variable
    generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))

    # Add noise to the variable
    noise = tf.random.uniform(tf.shape(generated_image), -0.25, 0.25)
    generated_image.assign_add(noise)  # Use assign_add to add noise to the variable
    # Clip the values within the range [0.0, 1.0]
    generated_image.assign(tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0))

    return content_image, style_image, generated_image



def initalizer(content_path, style_path, save_images = False):

    content_image, style_image, generated_image = define_ContentAndStyleAndGenerateImage(content_path, style_path)


    vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + content_layer)

    content_target = vgg_model_outputs(content_image)  # Content encoder
    style_targets = vgg_model_outputs([style_image])     # Style encoder

    # Assign the content image to be the input of the VGG model.
    # Set a_C to be the hidden layer activation from the layer we have selected
    preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
    a_C = vgg_model_outputs(preprocessed_content)

    # Assign the input of the model to be the "style" image
    preprocessed_style =  tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
    a_S = vgg_model_outputs(preprocessed_style)

    if save_images:
        for i in range(6):
            os.mkdir(f"images{i+1}")
    return vgg_model_outputs, a_S, a_C, content_image, style_image, generated_image

@tf.function()
def train_step(generated_image):
    with tf.GradientTape() as tape:
        # In this function you must use the precomputed encoded images a_S and a_C
        # Compute a_G as the vgg_model_outputs for the current generated image
        a_G = vgg_model_outputs(generated_image)

        # Compute the style cost
        J_style = compute_style_cost(a_S, a_G)
        print(J_style)
        # Compute the content cost
        J_content = compute_content_cost(a_C, a_G)
        # Compute the total cost
        J = total_cost(J_content, J_style)


    grad = tape.gradient(J, generated_image)

    optimizer.apply_gradients([(grad, generated_image)])
    generated_image.assign(clip_0_1(generated_image))
    # For grading purposes
    return J
if __name__ == "__main__":

    save_image = False
    vgg_model_outputs, a_S, a_C, content_image, style_image, generated_image = initalizer("D:/Cam graduate/DSC00412.JPG", "D:/Cam graduate/DSC00412.JPG", save_image)

    print("Defining variables Done..")

    epochs = 100000
    file =1
    images = 0
    for i in tqdm.tqdm(range(epochs+1)):
        C = train_step(generated_image)

        if not(save_image):
            continue

        if images == 2000:
            shutil.make_archive(f'/images{file}', 'zip', f'/images{file}')
            shutil.rmtree(f'/images{file}')
            file+=1
            images = 0
        if i % 10 == 0:
            image = tensor_to_image(generated_image)
            image.save(f"/images{file}/image-{i}-Cost:{C}.jpg")
            images +=1

        # if i % 5000 == 0:
        #     image = tensor_to_image(generated_image)
        #     print(f"Epoch {i} THE COST IS: {C}")
        #     imshow(image)
        #     plt.axis('off')
        #     plt.show()

    if save_image:
        shutil.make_archive(f'/images6', 'zip', f'/images6')
