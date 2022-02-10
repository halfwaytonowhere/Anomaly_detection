import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data_sheet = pd.read_csv( r"C:\Users\agata\OneDrive\Pulpit\studia\3 rok\DESIGNLAB\DL_project\serverlogs3.csv",sep=";")

data_sheet=data_sheet.drop(['Date first seen','Proto','Src IP Addr','Src Pt','Dst IP Addr','Dst Pt','BYTES2','Bytes','R','Packets','attackType','attackID','attackDescription','Flows','Tos','Flags','x','A','P','0x','F','S','class','czy M?'],axis=1, errors = 'ignore' ) 
print(data_sheet)
data_= data_sheet.values
                        # 1 is normal rythm and 0 is abnormal rythm which is label named unknow and suspicious
labels = data_[:, -1]   # getting the righttest cloumn turn into the labels  
data = data_[:, 0:-1]   # rest data 

trainData, testData, trainLabels, testLabels = train_test_split(
    data, labels, test_size=0.2, random_state=21)

# Normalizing datas by getting the mean, getting the max and then dividing the data minus the min by the max
# minus the min, to get basic normalization of the data. 

minimum_value = tf.reduce_min(trainData)                                         
maximum_value = tf.reduce_max(trainData)
trainData = (trainData - minimum_value) / (maximum_value - minimum_value)
testData = (testData - minimum_value) / (maximum_value - minimum_value)
trainData = tf.cast(trainData, tf.float32)
testData = tf.cast(testData, tf.float32)
trainLabels = trainLabels.astype(bool)
testLabels = testLabels.astype(bool)
normalTrainData = trainData[trainLabels]
normalTestData = testData[testLabels]
suspiciousTrainData = trainData[~trainLabels]
suspiciousTestData = testData[~testLabels]

class AnomalyDetector(Model):										#class AnomalyDetector based on the model from tensorflow.keras.models
  def __init__(self):												
    super(AnomalyDetector, self).__init__()			#Encoder its taking our datas and it given us dense represention of values of the data													
    self.encoder = tf.keras.Sequential([      	
      layers.Dense(64, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(16, activation="relu"),
      layers.Dense(8, activation="relu")
      ])

    self.decoder = tf.keras.Sequential([				#Decoder is going to reinflate the values back

      layers.Dense(32, activation="relu"),
      layers.Dense(64, activation="relu"),
      layers.Dense(128, activation="relu"),
      layers.Dense(256, activation="relu")
      ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded 

autoencoder = AnomalyDetector()									#defining our anomaly detector. 
autoencoder.compile(optimizer='adam', loss='mae')

history = autoencoder.fit(normalTrainData, normalTrainData, # Here we re gonna train it so its going to try to fit the normal training data 
           epochs=20,                                       # to the normal training data and its going to trainning 20 epochs
           batch_size=1024,
           validation_data=(testData, testData),
           shuffle=True)


# HEre we wanted to calculated the loss of the training data, and its plotting out loss as a histogram.
reconstructions = autoencoder.predict(normalTrainData)
train_loss = tf.keras.losses.mae(reconstructions, normalTrainData)

reconstructions = autoencoder.predict(suspiciousTestData)
test_loss = tf.keras.losses.mae(reconstructions, suspiciousTestData)

#Displaying characteristics to compare results  
fig, axs = plt.subplots(2)                 
axs[0].hist(train_loss[None,:], bins=50)
axs[0].set_title("Train loss")
axs[1].hist(test_loss[None, :], bins=50)
axs[1].set_title("Test loss")
plt.show()

threshold = np.mean(train_loss) + np.std(train_loss)     #checking a threshold
print("Threshold: ", threshold)

#Additional results displaying
def predict(model, data, threshold):
  reconstructions = model(data)
  loss = tf.keras.losses.mae(reconstructions, data)
  return tf.math.less(loss, threshold)

def print_stats(predictions, labels):
  print("Accuracy = {}".format(accuracy_score(labels, predictions)))

preds = predict(autoencoder, testData, threshold)
print_stats(preds, testLabels)