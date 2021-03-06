import os, argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import IPython.display as display
from PIL import Image
import pathlib
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

AUTOTUNE = tf.data.experimental.AUTOTUNE
DS_SIZE = 918
BATCH_SIZE = 128
IMG_SIZE = 256
IMG_HEIGHT = 256
IMG_WIDTH = 256
CLASS_NAMES = None

#-----------------LOAD DATA----------------------#
def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    if (parts[-2] == CLASS_NAMES[0]):
        return 0
    else:
        return 1

def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])

def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

def load_data(data_dir):
    global CLASS_NAMES

    data_dir = pathlib.Path(data_dir)

    classes = []
    for i in np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]):
        if i=='.DS_Store':
            continue
        else:
            classes.append(i)
    CLASS_NAMES = np.array(classes)

    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    return labeled_ds
#------------------------------------------------#

#------------PREPARE FOR TRAINING----------------#

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('sample.png')

def split_dataset_and_prepare(ds, cache=True, shuffle_buffer_size=1000):
    train_size = int(0.7 * DS_SIZE)
    val_size = int(0.15 * DS_SIZE)
    test_size = int(0.15 * DS_SIZE)

    train_dataset = ds.take(train_size)
    # train_dataset = train_dataset.repeat()
    test_dataset = ds.skip(train_size)
    val_dataset = ds.skip(val_size)
    test_dataset = ds.take(test_size)

    train_batches = train_dataset.shuffle(shuffle_buffer_size).batch(BATCH_SIZE)
    validation_batches = val_dataset.batch(BATCH_SIZE)
    test_batches = test_dataset.batch(BATCH_SIZE)

    return train_batches, validation_batches, test_batches

#------------------------------------------------#

#------------LOAD MODEL FROM BASE----------------#

def load_model():
    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
    # base_learning_rate = 0.0001
    # model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
    #           loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #           metrics=['accuracy'])
              
    return model

#------------------------------------------------#

#------------------TRAIN-------------------------#

def train_model(model, train_batches, validation_batches, test_batches, model_name, save=False):
    initial_epochs = 30
    validation_steps = 20
    
    # loss0, accuracy0 = model.evaluate(validation_batches, steps=validation_steps)

    # print("initial loss: {:.2f}".format(loss0))
    # print("initial accuracy: {:.2f}".format(accuracy0))

    total_train = len(list(train_batches)) * BATCH_SIZE
    total_val = len(list(validation_batches)) * BATCH_SIZE

    history = model.fit(train_batches, epochs=initial_epochs, validation_data=validation_batches)

    loss1, accuracy1 = model.evaluate(test_batches, steps=validation_steps)
    print("new loss: {:.2f}".format(loss1))
    print("new accuracy: {:.2f}".format(accuracy1))

    if save:
        model.save(model_name)

    return history

def train_model_fine(base_model, model, history, train_batches, validation_batches, test_batches, model_name, save=False):
    initial_epochs = 20
    validation_steps = 20

    base_model.trainable = True
    fine_tune_at = 100

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable =  False
    
    base_learning_rate = 0.0001
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])
    
    fine_tune_epochs = 20
    total_epochs =  initial_epochs + fine_tune_epochs
    
    history_fine = model.fit(train_batches, epochs=total_epochs, initial_epoch=history.epoch[-1], validation_data=validation_batches)

    loss1, accuracy1 = model.evaluate(test_batches, steps=validation_steps)
    print("new loss: {:.2f}".format(loss1))
    print("new accuracy: {:.2f}".format(accuracy1))

    if save:
        model.save('saved_model/' + model_name)

#------------------------------------------------#

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../images",
        help="Directory containing folders of images of each category"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="new_model",
        help="Name of saved model"
    )
    FLAGS = parser.parse_args()

    labeled_ds = load_data(FLAGS.data_dir)

    train_batches, validation_batches, test_batches = split_dataset_and_prepare(labeled_ds)
    
    model = load_model()
    
    history = train_model(model, train_batches, validation_batches, test_batches, 'saved_model/' + FLAGS.model_name, save=True)
    
    # train_model_fine(base_model, model, history, train_batches, validation_batches, test_batches, FLAGS.model_name, save=True)