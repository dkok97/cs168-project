# TRANSFER LEARNING ON RENAL SCANS

## Directory Structure
1. src/: all the code
    1. saved_model/: model saved from last training
    2. create_pathology.py: used to generate pathology.csv
    3. train.py: used for creating, training, and saving the model
2. pathology.csv: labels for all patients
3. requirements.txt: library requirements

## How to run

* Create a virtual environment and activate it (Use python 3)
```
virtualenv venv
source venv/bin/activate
```

* Install requirements
```
pip3 install -r requirements.txt
```

* Create an images directory (in the root of the folder) with the following structure:
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

* Go to src/ and run the following command with appropriate parameters
```
python3 train.py --data_dir=../images --model_nane={name to save model with}
```
Model with be saved at src/saved_model/{model_name}