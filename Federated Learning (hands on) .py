
# coding: utf-8

# In[87]:


get_ipython().system('pip install -q tensorflow==2.0.0-beta1')
get_ipython().system('pip install --quiet  tf-nightly')
get_ipython().system('pip install -q h5py pyyaml')


# In[88]:


import tensorflow as tf
from tensorflow.keras import datasets, models, layers
import numpy as np


# In[89]:


''' 
Loading the MNIST dataset
''' 

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

train_images, train_labels = train_images[:30000], train_labels[:30000]
test_images, test_labels = test_images[:9000], test_labels[:9000]
train_images, test_images = train_images/255.0, test_images/255.0


# In[97]:


''' 
Building and compiling the model, considered loss as sparse_categorical_crossentropy since our output is an integer.
Conv2D(32) -> MaxPooling2D -> Conv2D(64) -> MaxPooling2D -> Flatten(3D to 1D) -> Dense(64) -> Dense(10)
''' 

def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# In[98]:


''' 
Training the model over the given parameters.
''' 

def fit_model(model, train_images, train_labels, epochs):
    model.fit(train_images, train_labels, epochs=epochs)
    return model


# In[99]:


''' 
Evaluating and printing the accuracy.
''' 

def test_model(model, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(test_acc)


# In[100]:


'''
Creating three kind of dataset, each having half the dataset of the overall one.
'''

train_images_1, train_labels_1 = train_images[:5000], train_labels[:5000]
train_images_2, train_labels_2 = train_images[5000:10000], train_labels[5000:10000]
train_images_3, train_labels_3 = train_images[10000:15000], train_labels[10000:15000]

test_images_1, test_labels_1 = test_images[:3000], test_labels[:3000]
test_images_2, test_labels_2 = test_images[3000:6000], test_labels[3000:6000]
test_images_3, test_labels_3 = test_images[6000:9000], test_labels[6000:9000]


# In[101]:


def get_compiled_model(train_images, train_labels, epochs):
    model = build_model()
    model = fit_model(model, train_images, train_labels, epochs)
    return model


# In[102]:


''' 
Picking first 5 data to train the model, such that it has initial weights.
''' 

initial_train_images = train_images[:5]
initial_train_labels = train_labels[:5]

print(initial_train_images.shape, initial_train_labels.shape)


# In[103]:


'''
Initial model is being trained, this model would be serialized and sent over to client
where it would be retrained on different dataset and sent back to the server.
'''

initial_model = get_compiled_model(initial_train_images, initial_train_labels, 5)
initial_model.save('initial_model.h5')


# In[104]:


''' 
Testing the initial model, initial accuracy of 20% approx.
'''

test_model(initial_model, test_images, test_labels)


# In[105]:


''' 
Considering we have two clients, each receive the same initial model.
''' 

initial_model_1 = models.load_model('initial_model.h5')
initial_model_2 = models.load_model('initial_model.h5')
initial_model_3 = models.load_model('initial_model.h5')


# In[106]:


''' 
Training the initial model over the new dataset, we won't be compiling the model, only fit the model.
''' 

# Client 1:
initial_model_1.fit(train_images_1, train_labels_1, epochs=2)

# Client 2:
initial_model_2.fit(train_images_2, train_labels_2, epochs=2)

# Client 3:
initial_model_3.fit(train_images_3, train_labels_3, epochs=2)


# In[108]:


'''
Testing the models over entire dataset, The highest accuracy obtained is 95.90% and the lowest is 95.29%.
'''
test_model(initial_model_1, test_images, test_labels)
test_model(initial_model_2, test_images, test_labels)
test_model(initial_model_3, test_images, test_labels)


# In[109]:


'''
Averaging the weights as done in the paper:
Communication-Efficient Learning of Deep Networks from Decentralized Data - H.Brendan McMahan, 
Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agu ̈era y Arcas.
'''
resultant_weights = np.add(initial_model_1.weights, initial_model_2.weights)
resultant_weights = np.add(resultant_weights, initial_model_3.weights)
resultant_weights = resultant_weights/3


# In[110]:


'''
Loading the initial model and testing on the overall test data. The accuracy obtained is 15%, 
Next we would set the averaged weights from two models and test it again.
'''

resultant_model = models.load_model('initial_model.h5')
test_model(resultant_model, test_images, test_labels)


# In[111]:


'''
Setting the averaged weights and testing on the overall test data. The accuracy obtained is 96.44%.
'''

resultant_model.set_weights(resultant_weights)
test_model(resultant_model, test_images, test_labels)

