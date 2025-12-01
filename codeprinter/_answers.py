"""Central answer store for the codeprinter package."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Union

Answer = Union[str, int, float, dict[str, Any], list[Any], Callable[[], Any]]

ANSWERS: dict[int, Answer] = {
    1: ('''Question 1:
Create and implement a basic neuron model within a computational framework.
Integrate essential elements like input nodes, weight parameters, bias, and an
activation function (including but not limited to sigmoid, hyperbolic tangent, and
Rectified Linear Unit (ReLU)) to construct a comprehensive representation of a
neuron. Evaluate and observe how each activation function influences the network's
behaviour and effectiveness in handling different types of data.

Answer 1:
#Lab-01
import numpy as np
import matplotlib.pyplot as plt
class Neuron:
  def __init__(self, input_size):
    self.weights = np.random.rand(input_size)
    self.bias = np.random.rand()
  
  def sigmoid(self, x):
      return 1 / (1 + np.exp(-x))
  def tanh(self, x):  
      return np.tanh(x)
  def relu(self, x):
      return np.maximum(0,x)
  def softmax(self, x):   
      return np.exp(x)/np.sum(np.exp(x))
  
  def forward(self, inputs):
      z = np.dot(inputs, self.weights) + self.bias
      return self.sigmoid(z)

input_size = 5
inputs = np.random.rand(10, input_size) 
neuron = Neuron(input_size)
for i in range(len(inputs)):
  print(f"Output {i}: {neuron.forward(inputs[i])}")

#Sigmoid graph
x = np.linspace(-10, 10, 100)
sigmoid_values = neuron.sigmoid(x)
plt.plot(x, sigmoid_values)
plt.title('Sigmoid Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()

#Hyperbolic tan graph
t = np.linspace(-5, 5,1000)
hyperbolictanh_values = neuron.tanh(t)
plt.plot(t, hyperbolictanh_values, )
plt.title('Hyperbolic Tan Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()

#Relu graph
r = np.linspace(-5,5)
relu_values = neuron.relu(r)
plt.plot(r, relu_values)
plt.title('Relu Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()

#Softmax graph
s = np.linspace(-5,5)
softmax_values = neuron.softmax(s)
plt.plot(s, softmax_values)
plt.title('Softmax Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()

'''),
    2: ('''Question 2:
Develop and implement a program to execute the perceptron learning algorithm,
customized to train a single-layer perceptron for binary classification tasks. create a
robust algorithm that refines the model's weights iteratively, resulting in a proficient
single-layer perceptron capable of effectively handling binary classification
challenges.

Answer 2:
#LAB-02
import numpy as np
class Perceptron:
  def __init__(self, learning_rate=0.01, epochs=100):
    self.lr = learning_rate
    self.epochs = epochs
    self.weights = None
    self.bias = 0
  def fit(self, X, y):
    self.weights = np.zeros(X.shape[1])

    for epoch in range(self.epochs):
      for i in range(X.shape[0]):
        linear_output = np.dot(X[i], self.weights) + self.bias
        y_pred = 1 if linear_output >= 0 else 0

        error = y[i] - y_pred

        self.weights += self.lr * error * X[i]
        self.bias += self.lr * error
        print(f'Epoch {epoch+1}, Sample {i+1}: Weights: {self.weights}, Bias: {self.bias}')

  def predict(self, x):
      linear_output = np.dot(x, self.weights) + self.bias
      return 1 if linear_output >= 0 else 0

X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([0, 0, 0, 1])

print("Training Perceptron on AND Gate...")
model = Perceptron(learning_rate=0.1, epochs=10)
model.fit(X_train, y_train)

test_point = np.array([1, 1])
prediction = model.predict(test_point)
print(f"\nPrediction for {test_point}: {prediction}")

'''),
    3:('''
Question 3:
Implement a program aimed at constructing and training a multilayer perceptron
tailored for a specific task, showcasing the execution of the backpropagation
algorithm. Construct a network with a linear input layer, Tanh or ReLU activation for
the hidden layers, and sigmoid or SoftMax activation for the output layer.

Answer 3:

#LAb-03
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential([
  Dense(64, activation='relu', input_shape=(4,)),
  Dense(32, activation='relu'),
  Dense(3, activation='softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Training MLP...")
history = model.fit(X_train, y_train, epochs=50, validation_split=0.1,verbose=0)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
    '''),
    4: ('''
Question 4:
Design a deep NN and optimize the network with Gradient Descent and optimize the
same with Stochastic gradient descent (SGD).

Answer 4:

#Lab-04
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt

iris = load_iris()
x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


def run_experiment(batch_size, label_name):
  model=Sequential([
      Dense(64,activation='relu',input_shape=(4,)),
      Dense(32, activation='relu'),
      Dense(3,activation='softmax')
  ])
  
  model.compile(optimizer=SGD(learning_rate=0.01),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
  history = model.fit(X_train, y_train, epochs=20,batch_size=batch_size, verbose=0, validation_split=0.1)

  return history

print("Running Gradient Descent...")
history_gd = run_experiment(len(X_train), 'Gradient Descent (Batch=All)')

print("Running Stochastic Gradient Descent...")
history_sgd = run_experiment(1, 'Stochastic GD (Batch=1)')

# Plot 1: Loss (Train vs Val) for GD and SGD
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_gd.history['loss'], label='Train Loss')
plt.plot(history_gd.history['val_loss'], label='Val Loss')
plt.title('Batch GD Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history_sgd.history['loss'], label='Train Loss')
plt.plot(history_sgd.history['val_loss'], label='Val Loss')
plt.title('SGD Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot 2: Accuracy (Train vs Val) for GD and SGD
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_gd.history['accuracy'], label='Train Acc')
plt.plot(history_gd.history['val_accuracy'], label='Val Acc')
plt.title('Batch GD Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history_sgd.history['accuracy'], label='Train Acc')
plt.plot(history_sgd.history['val_accuracy'], label='Val Acc')
plt.title('SGD Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot 3: GD vs SGD Loss
plt.figure(figsize=(10, 6))
plt.plot(history_gd.history['loss'], label='Gradient Descent')
plt.plot(history_sgd.history['loss'], label='Stochastic GD')
plt.title('Gradient Descent vs SGD Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

    '''),
    5:('''
Question 5:
Demonstrate the usage of dropout, gradient clipping and multitask learning with early
stopping in a neural network training scenario.
   
Answer 5:
#Lab-05
import numpy as np, matplotlib.pyplot as plt, tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import make_classification, load_digits
from sklearn.model_selection import train_test_split

# Dropout vs Clipping
X, y = make_classification(1000, 20, n_classes=2)
X_tr, X_te, y_tr, y_te = train_test_split(X, y)

def train_model(name, opt, use_drop=False):
    model = Sequential()
    model.add(Dense(64, 'relu', input_shape=(20,)))
    if use_drop: model.add(Dropout(0.2)) 
    model.add(Dense(1, 'sigmoid'))

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    h = model.fit(X_tr, y_tr, epochs=20, validation_data=(X_te, y_te), verbose=0)
    plt.plot(h.history['val_accuracy'], label=name)


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("Dropout vs Clipping")
train_model("Dropout", Adam(), use_drop=True)
train_model("Clipping", Adam(clipnorm=1.0))
plt.legend()


# Multitask & Early Stopping
d = load_digits()
X, y = d.images/16.0, d.target
y_par = y % 2 
X_tr, X_te, y_tr, y_te, yp_tr, yp_te = train_test_split(X, y, y_par)

inp = Input((8,8))
z = Flatten()(inp)
z = Dense(32, 'relu')(z)
out_digit = Dense(10, 'softmax')(z) 
out_parity = Dense(1, 'sigmoid')(z) 
model = Model(inp, [out_digit, out_parity])

model.compile(optimizer=Adam(), loss=['sparse_categorical_crossentropy', 'binary_crossentropy'])
es = EarlyStopping(monitor='val_loss', patience=3)
h = model.fit(X_tr, [y_tr, yp_tr], epochs=30, callbacks=[es],
validation_split=0.2, verbose=0)

plt.subplot(1,2,2)
plt.title("Early Stopping Loss")
plt.plot(h.history['loss'], label='Total Loss')
plt.plot(h.history['val_loss'], label='Val Loss')
plt.legend()
plt.show()

    '''),
    6:('''
Question 6:
Implement a program on Adversarial training, tangent distance, tangent prop and
tangent classifier. [Any three to be implemented].

Answer 6:

#Lab-06
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2,random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

def augment_data(data, labels, epsilon, mode):
  if epsilon == 0:
      return data, labels

  noise = np.random.randn(*data.shape)
  perturbation = None # Initialize perturbation

  if mode == 'sign':
      perturbation = np.sign(noise)
  elif mode == 'normal': # Added condition for 'normal' mode
      perturbation = noise
  else:
      raise ValueError(f"Unknown augmentation mode: {mode}")

  x_aug = data + epsilon * perturbation

  return np.concatenate([data, x_aug]), np.concatenate([labels,labels])

experiments = {
"Adversarial Training": (0.1, 'sign'),
"Tangent Prop": (0.1, 'normal'),
"Tangent Classifier": (0.01, 'normal'),
"Tangent Distance": (0.0, None)
}

history_dict = {}
results = {}
print("Training Models...")
for name, (eps, mode) in experiments.items():
  x_tr_aug, y_tr_aug = augment_data(X_train, y_train, eps, mode)

  model = Sequential([
    Dense(64, activation='relu', input_shape=(20,)),
    Dense(1, activation='sigmoid')
  ])
  model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

  print(f" -> Training {name}...")
  hist = model.fit(x_tr_aug, y_tr_aug, epochs=20, batch_size=32,verbose=0)
  history_dict[name] = hist

  loss, acc = model.evaluate(X_test, y_test, verbose=0)
  results[name] = (loss, acc)

print("\n--- Final Results ---")
for name, (loss, acc) in results.items():
  print(f"{name}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
for name, hist in history_dict.items():
  plt.plot(hist.history['loss'], label=name)
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
for name, hist in history_dict.items():
  plt.plot(hist.history['accuracy'], label=name)
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

    '''),
    7:('''
    
    Question 7:
Develop a program to classify the MNIST Dataset using Convolution Neural Network
[CNN].

Answer 7:
#Lab-7
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten,Dense
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential([
  Conv2D(32, kernel_size=(3,3), activation='relu',input_shape=(28,28,1)),
  MaxPooling2D(pool_size=(2,2)),
  Flatten(),
  Dense(128, activation='relu'),
  Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
print("Training CNN...")
history = model.fit(x_train, y_train, epochs=3, batch_size=64,validation_split=0.1)
predictions = model.predict(x_test)
predicted_labels= [tf.argmax(prediction).numpy() for prediction in predictions]

plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('CNN Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('CNN Loss')
plt.legend()
plt.show()

plt.figure(figsize=(10, 4))
num_images = 3
for i in range(num_images):
  plt.subplot(1,num_images, i + 1)
  plt.imshow(x_test[i].reshape(28,28), cmap="gray")
  plt.title(f"Predicted: {predicted_labels[i]}, True:{tf.argmax(y_test[i]).numpy()}")
  plt.axis('off')
  plt.show()

    '''),
    8:('''
Question 8:
Design a python program for Sentiment Analysis using Recurrent Neural Networks
(RNN).

Answer 8:

#Lab-08
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

max_features = 10000
maxlen = 100
print("Loading IMDB data...")
(x_train, y_train), (x_test, y_test) =imdb.load_data(num_words=max_features)

x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

model = Sequential([
    Embedding(max_features, 50, input_length=maxlen),
    SimpleRNN(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5, batch_size=128,validation_data=(x_test, y_test), verbose=2)

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print("Test Accuracy:", acc)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1);
plt.plot(history.history['accuracy']);
plt.plot(history.history['val_accuracy']);
plt.title("Accuracy")

plt.subplot(1,2,2);
plt.plot(history.history['loss']);
plt.plot(history.history['val_loss']);
plt.title("Loss")
plt.tight_layout();
plt.show()

    '''),
    9:('''
Develop a GRU based term stock price prediction model for tickers in yahoo finance.

Answer 9:

#Lab-09
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

df = yf.download("AAPL", period="5y")["Close"].values.reshape(-1,1)
mn, mx = df.min(), df.max()
data = (df - mn) / (mx - mn)

X, y = [], []
seq_len = 60
for i in range(len(data) - seq_len):
    X.append(data[i : i+seq_len])
    y.append(data[i+seq_len])

X, y = np.array(X), np.array(y)

model = Sequential([
    GRU(64, input_shape=(seq_len, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=5, batch_size=32, verbose=1)


pred = model.predict(X) * (mx - mn) + mn
actual = y * (mx - mn) + mn
plt.plot(actual, label="Actual Price")
plt.plot(pred, label="Predicted Price")
plt.legend()
plt.show()

    '''),
    10:('''
Implement stock market prediction using Long Short-Term Memory (LSTM).

Answer 10:
#LAb-10
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

df = yf.download("AAPL", period="5y")["Close"].values.reshape(-1,1)

mn, mx = df.min(), df.max()
data = (df - mn) / (mx - mn)

X, y = [], []
seq_len = 60 
for i in range(len(data) - seq_len):
    X.append(data[i : i+seq_len])
    y.append(data[i+seq_len])

X, y = np.array(X), np.array(y)

model = Sequential([
    LSTM(64, input_shape=(seq_len, 1)),
    Dense(1) 
])

model.compile(optimizer='adam', loss='mse')
print("Training LSTM...")
model.fit(X, y, epochs=5, batch_size=32, verbose=1)

pred = model.predict(X) * (mx - mn) + mn
actual = y * (mx - mn) + mn
plt.plot(actual, label="Actual Price")
plt.plot(pred, label="Predicted Price")
plt.title("Stock Prediction (LSTM)")
plt.legend()
plt.show()
    ''')
}

