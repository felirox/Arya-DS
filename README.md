# Data Scientist Assignment

#### We are developing a Binary Classifier using TensorFlow and Keras with the [following dataset.](https://drive.google.com/file/d/12GAAr58y1bI1vTWknR4MXOxeXHtjZItl/view) 

You can clone this repo by running
```bash
git clone https://github.com/felirox/Arya-DS
```

## Installation

The program was written using Python3. The code is available in a Python Notebook (.ipynb) file. 

No additional requirement is necessary if you have Python/Jupyter installed. 

To download Anaconda/Jupyter, [click here](https://www.anaconda.com/)

## Requirements

To install the necessary requirements, open your terminal and head over to the directory where this cloned code is present. 

Run the following command to install all the necessary libraries:

```bash
pip install -r requirements.txt
```
## Download on Google Colab 

You can run the same code online via Google Colab without having to download or install any of the above mentioned items. 

#### [Click here](https://bit.ly/3kb91FX) to open the notebook in Google Colab.

## Thought Process and General Approach

- The dataset is download and the csv files are extracted from the zip folder. Pandas is used to make a dataframe using the given data. 
- The given data is shuffled using the below command. frac=1 allows us to shuffle the data. This ensures consistent data and doesn't allow clumping, if any.
   ```python
   data.sample(frac=1).reset_index(drop=True)
   ```
- The training set is split in a ratio of 4:1 for Training Set and Validation Set. 
- Training set is utilized while training and Validation set helps us in seeing the effectives on unknown data in real-time
- On viewing the dataset, we can see that the data is **not standardized.** This would make it hard for the DNN to understand the input parameters as the difference between the input values are too high and the model does not perform at it's best under such circumstances. 
- To solve this, we will be standardizing the dataset using *Standard scaling*
- Multiple versions of the model were created with various levels of hidden layers and nodes. These were tuned based on the performance analysis of the model and it's training and validation scores range for n epochs. 
- The final version of the Sequential layers used are 
   ```python
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(1, activation='sigmoid')
  ])
   ```
- #### Parameters used in the Model:
     - **Optimizer:** Adam 
     - **Learning Rate:** 0.0001 *(Can be varied by trading off with the number of hidden layers)*
     - **Loss:** Binary Crossentropy
     - **No. of Epochs trained:** 100 *(Significant decrement in Training Loss - Validation Loss decreases until 30 epochs)  

## Model Evaluation Metrics

The below graph presents the Accuracy, Precision, Recoil and Loss of both Training and Validation dataset

## Model Predictions for Test Dataset

The file containing the model predictions for the given test dataset can be found here. The output is saved in .txt format with the regular UTF-8 encoding.

We have converted the Predicted Probabilities into Binary Class labels with a threshold of 0.7.

### View the Prediction file here: [op.txt]()

You can view the Actual Probabilities which have not been filtered by a threshold, [here.]()

<img width="521" alt="final-model metrics" src="https://user-images.githubusercontent.com/52323747/165163550-c65528f1-fbad-4dc7-be9d-a90c7ad92829.png">

## Libraries and Dependencies

- Python 3.8.8 64-bit and Jupyter Notebook (Anaconda 2021.05)

- ### Libraries Used: 

   ```python
   import pandas as pd
   from keras.models import Sequential
   from keras.layers import Dense
   from keras.wrappers.scikit_learn import KerasClassifier
   from sklearn.model_selection import cross_val_score
   from sklearn.preprocessing import LabelEncoder
   from sklearn.model_selection import StratifiedKFold
   from sklearn.preprocessing import StandardScaler
   from sklearn.pipeline import Pipeline
   import matplotlib.pyplot as plt
   from matplotlib import rcParams
   import numpy as np
   from sklearn.preprocessing import StandardScaler
   import tensorflow as tf
   import os
   ```



## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)

Other license are applicable as per the dataset owners
