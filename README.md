# TRANSFER LEARNING ON RENAL SCANS

## Directory Structure

Train, Validatation, and Test images were omitted from the repo due to their large size.
These images were generated using the scripts in this [repository](https://github.com/kahsieh/renal-lesion-classifier-mirror)

1. src/: all the code
    1. saved_model/: Weights and Models saved from training (see name for description)
    2. create_pathology.py: used to generate pathology.csv for all patients
    3. cs168_train.py: used for training and testing a Keras Sequential model. Data is created using tf.keras.preprocessing.
    4. cs168_train_transfer.py: used for training and testing a Keras Sequential model using transfer learning on top of the Inception V3 architecture
    5. old-train.py: used for creating, training, and saving a tensorflow CNN using transfer learaning on top of the Mobilenet architecture
    6. old-train_nopretrain.py: used for training and testing a Keras Sequential model. Data is created using tf.data.Dataset.
2. pathology.csv: labels for all patients
3. requirements.txt: library requirements
4. report-images: graphs created during training and testing

## How to run

- Create a virtual environment and activate it (Use python 3)
```
virtualenv venv
source venv/bin/activate
```

- Install requirements
```
pip3 install -r requirements.txt
```

- Create an images directory (in the root of the folder) with the following structure:
```
images
└───BENIGN
│   │   image1.jpeg
│   │   image2.jpeg
│   │   ...
└───MALIGNANT
│   │   image1.jpeg
│   │   image2.jpeg
│   │   ...
```

- Go to src/ and run the following command with appropriate parameters
```
python3 old-train.py --data_dir=../images --model_nane={name to save model with}
```
Model with be saved at src/saved_model/{model_name}

- If you wish to run the cs168_ python scripts (improved and optimized), open them using Google Colab and set runtime processor to GPU. This would require you to upload the training and testing data to your google drive so as to give the python notebook access to the data.