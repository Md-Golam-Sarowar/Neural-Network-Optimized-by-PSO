import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from nn import sigmoid, tanh
from nn.model import Model
from nn.layers import Layer
from nn.losses import BinaryCrossEntropyLoss
from nn.pipeline import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import timeit
import time
from timeit import default_timer as timer
import os
import psutil


start = timeit.timeit()
start1 = time.time()
start2 = timer()
t = time.process_time()

data = pd.read_csv("Final_data.csv")
#shuffling datasets
data = data.sample(frac=1)

x = data.iloc[:, 0:4]
y = data['Significance']


total=0
n = 30044

for i in range(len(x)):
    total = total+x[i]

m = np.sqrt((total)/n)

print(m)

encode = LabelEncoder()
y = encode.fit_transform(y)

all_data = np.array(x)

#binary encoded (one hot encoder)
Onehot_encoder = OneHotEncoder(sparse=False)
all_data = all_data.reshape(len(all_data), 1)
encoded_x = Onehot_encoder.fit_transform(all_data)

x_train, x_test, y_train, y_test = train_test_split(encoded_x, y, test_size= 0.15, random_state=21)

def accuracy(y, y_hat):
    y_hat = (y_hat >= 0.5).astype('int')
    y = y.astype('int')
    return np.mean(y_hat[:, 0] == y)

model = Model()
model.add_layer(Layer(965, 10, tanh))
model.add_layer(Layer(10, 10, tanh))
model.add_layer(Layer(10, 10, tanh))
model.add_layer(Layer(10, 10, tanh))
model.add_layer(Layer(10, 1, sigmoid))

model.compile(BinaryCrossEntropyLoss, DataLoader, accuracy, batches_per_epoch=20, n_workers=10)
print(x_train.shape, y_train.shape, y_train.shape, y_test.shape)
index_list, cost_list = model.fit(x_train, y_train, 500)
y_hat = model.predict(x_test)
#print(confusion_matrix(y_test, y_hat))

plt.plot(index_list, cost_list)
plt.xticks(index_list, rotation='vertical')
plt.xlabel("Number of Iterarion")
plt.ylabel("Cost")
plt.show()

end = timeit.timeit()
end1 = time.time()
end2 = timer()
elapsed_time = time.process_time() - t

print(end - start, "    ", end1 - start1, "      ", end2 - start2, "     ", elapsed_time )
print('Accuracy on test:', accuracy(y_test, y_hat))

process = psutil.Process(os.getpid())
print(process.memory_info().rss)
#display_top(end6)
