#Importing necessary libraries

import re
import scipy
import pandas as pd
import io
import numpy as np
import copy

import torch

from sklearn.metrics import classification_report
from sklearn.feature_extraction.text  import TfidfVectorizer

from torch import nn, optim
from torch.utils import data


from Data_Cleaning_functions import processStanceData

#Seeding for deterministic results
RANDOM_SEED = 16
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
HIDDEN_LAYER_UNITS = 128

CLASS_NAMES = ['support', 'deny', 'query', 'comment']
EPOCHS      = 50



#Reading Twitter and Reddit data (train, dev and test) as dataFrames
twitterTrainDf = pd.read_csv('TwitterTrainDataSrc.csv')
redditTrainDf  = pd.read_csv('RedditTrainDataSrc.csv')

twitterDevDf   = pd.read_csv('TwitterDevDataSrc.csv')
redditDevDf    = pd.read_csv('RedditDevDataSrc.csv')

twitterTestDf  = pd.read_csv('TwitterTestDataSrc.csv')
redditTestDf   = pd.read_csv('RedditTestDataSrc.csv')

#Processing Twitter and Reddit dataframe containig training data
trainDf = processStanceData(twitterTrainDf, redditTrainDf)
trainDf.drop(np.where(trainDf!= trainDf)[0][0], inplace=True) # this line contains Nan Value

devDf = processStanceData(twitterDevDf, redditDevDf)
testDf = processStanceData(twitterTestDf, redditTestDf)

x_train = trainDf['TextSrcInre'].tolist()
y_train = trainDf['labelValue'].tolist()


x_dev  = devDf['TextSrcInre'].tolist()
y_dev  = devDf['labelValue'].tolist()

x_test = testDf['TextSrcInre'].tolist()
y_test = testDf['labelValue'].tolist()

# Instantiating TfidfVectorizer object and fitting it on the training set
tfidf = TfidfVectorizer(min_df = 10, max_df = 0.5, ngram_range=(1,2))
_ = tfidf.fit(x_train)


class Tfidf_Nn(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden  = nn.Linear(len(tfidf.get_feature_names_out()), HIDDEN_LAYER_UNITS)
        # Output layer
        self.output  =  nn.Linear(HIDDEN_LAYER_UNITS, len(CLASS_NAMES))
        self.dropout = nn.Dropout(0.1)
        
        # Defining tanh activation and softmax output 
        self.tanh    = nn.Tanh()                                     #Using tanh as it performed better than ReLu during hyper-param optimisation
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Pass the input tensor through each of the below operations
        x = self.hidden(x)
        #print(x.shape)
        y = self.tanh(x)
        #print(y.shape)
        z = self.dropout(y)
        #print(z.shape)
        z = self.output(z)
        #print(z.shape)
        z = self.softmax(z)
        
        #returning the output from hidden layer and the output layer
        return  y, z
    

class StanceDetector:
  def __init__(self,model,tfidf,class_weights=[8.0, 20.0, 8.0, 1.0],optimizer=optim.Adam,learning_rate=0.02):
    self.model = model
    self.class_weights = torch.FloatTensor(class_weights)
    self.criterion = self.criterion_fn()
    self.optimizer = optimizer(model.parameters(), lr=learning_rate)
    self.tfidf = tfidf

  def criterion_fn(self):
    criterion = nn.CrossEntropyLoss(weight = self.class_weights)
    return criterion

  def fit(self,x_train,y_train,x_dev,y_dev,epochs=50,verbose=1):
    train_losses = []
    train_accuracies = []
    dev_losses = []
    dev_accuracies = []
    
    x_train_feats = self.tfidf.fit(x_train)
    x_train_transform = x_train_feats.transform(x_train)
    #Converting the TF-IDF matrix to tensor
    tfidf_transform_tensor = torch.tensor(scipy.sparse.csr_matrix.todense(x_train_transform)).float()
    #Tranforming the development and test data to tf-idf matrix
    x_dev  = self.tfidf.transform(x_dev)
    x_dev  = torch.tensor(scipy.sparse.csr_matrix.todense(x_dev)).float()
    
    #Converting prections for train, dev and test data to tensors
    y_train = torch.tensor(y_train)
    y_dev   = torch.tensor(y_dev)
    
    for e in range(epochs):
      correct_predictions = 0
      self.optimizer.zero_grad()
      
      
      hidden_layer_output, classifier_output = self.model.forward(tfidf_transform_tensor)

      loss = self.criterion(classifier_output, y_train.type(torch.LongTensor))
      loss.backward()
      train_loss = loss.item()
      train_losses.append(train_loss)
      
      #Calculating values predicted by the model
      _, preds = torch.max(classifier_output, dim=1)
      correct_predictions += torch.sum(preds == y_train)
      
      #Calculating accuracy
      train_accuracy = correct_predictions.double() / len(y_train)
      train_accuracies.append(train_accuracy)

      self.optimizer.step()
      correct_predictions = 0
      with torch.no_grad():
          self.model.eval()

          #Getting hidden layer and softmax output from model for dev data
          hidden_layer_output, classifier_output = self.model(x_dev)
          
          #Calculating loss
          dev_loss = self.criterion(classifier_output, y_dev)
          dev_losses.append(dev_loss)

          #Calculating values predicted by the model
          _, preds = torch.max(classifier_output, dim=1)
          correct_predictions += torch.sum(preds == y_dev)
          
          #Calculating accuracy
          dev_accuracy = correct_predictions.double() / len(y_dev)
          dev_accuracies.append(dev_accuracy)

      self.model.train()
      if verbose:
        print(f"Epoch: {e+1}/{epochs}.. ",
              f"Training Loss: {train_loss:.3f}.. ",
              f"Training Accuracy: {train_accuracy:.3f}",
              f"Dev Loss: {dev_loss:.3f}.. ",
              f"Dev Accuracy: {dev_accuracy:.3f}")
      
    return {
            'train_loss':train_losses,
            'train_acc':train_accuracies,
            'dev_loss':dev_losses,
            'dev_acc':dev_accuracies
            }



  def predict(self, x_test):
    predictions = []
    x_train_feats = self.tfidf.fit(x_train)
    x_test = self.tfidf.transform(x_test)

    x_test = torch.tensor(scipy.sparse.csr_matrix.todense(x_test)).float()
    #Converting prections for train, dev and test data to tensors
    with torch.no_grad():
      self.model.eval()
      _,classifier_output = self.model(x_test)
      _, preds = torch.max(classifier_output, dim=1)
      predictions.extend(preds)
    predictions = torch.stack(predictions)

    mapping = {
      0:'support',
      1:'deny',
      2:'query',
      3:'comment'
    }

    label_predictions = []
    for i in predictions:
      label_predictions.append(mapping[i.item()])
    return predictions,label_predictions

  def get_predictions(self, x_test, y_test):
    predictions = []
    prediction_probs = []
    real_values = []
    
    x_train_feats = self.tfidf.fit(x_train)
    x_test = self.tfidf.transform(x_test)

    x_test = torch.tensor(scipy.sparse.csr_matrix.todense(x_test)).float()
    y_test  = torch.tensor(y_test)
    with torch.no_grad():
      self.model.eval()
      labels = y_test

      #Currently, not interested in the hidden layer outputs.
      _,classifier_output = self.model(x_test)

      #Not interested in the maximum values, interested with the indices of these max values
      _, preds = torch.max(classifier_output, dim=1)

      predictions.extend(preds)
      prediction_probs.extend(classifier_output)
      real_values.extend(labels)
    predictions = torch.stack(predictions)

    prediction_probs = torch.stack(prediction_probs)
    real_values = torch.stack(real_values)
    return  predictions, prediction_probs, real_values

  def predict_new(self,model,filepath):
    with open(filepath) as f:
      lines = f.readlines()
    df = lines
    predictions = []
    x_testdf = tfidf.transform(df)

    x_testdf = torch.tensor(scipy.sparse.csr_matrix.todense(x_testdf)).float()
    #Converting prections for train, dev and test data to tensors
    with torch.no_grad():
      model.eval()
      _,classifier_output = model(x_testdf)
      _, preds = torch.max(classifier_output, dim=1)
      predictions.extend(preds)
    predictions = torch.stack(predictions)

    mapping = {
      0:'support',
      1:'deny',
      2:'query',
      3:'comment'
    }

    label_predictions = []
    for i in predictions:
      label_predictions.append(mapping[i.item()])
    return predictions,label_predictions

#stanceDetector = StanceDetector(model,tfidf) # the class takes in as arguments : the model, and the vectorizer
#history = stanceDetector.fit(x_train,y_train,x_dev,y_dev,epochs=100,verbose=1)