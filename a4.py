
# coding: utf-8

# In[19]:


import os, cv2, random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.utils import np_utils
import numpy as np


from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils


# In[20]:


from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import numpy
from keras.datasets import mnist


# In[21]:


s1= "Documents/Iris_Imgs"


# In[22]:


versicolor = "Documents/Iris_Imgs/versicolor"
virginica =  "Documents/Iris_Imgs/virginica"
setosa = "Documents/Iris_Imgs/setosa"


# In[23]:


train_batch = ImageDataGenerator().flow_from_directory(s1, target_size = (224,224), classes = ["versicolor", "virginica", "setosa"], batch_size = 10)
test_batch = ImageDataGenerator().flow_from_directory(s1, target_size = (224,224), classes = ["versicolor", "virginica", "setosa"], batch_size =10)
valid_batch = ImageDataGenerator().flow_from_directory(s1, target_size = (224,224), classes = ["versicolor", "virginica", "setosa"], batch_size = 10)


# In[24]:


# plots images with labels within jupyter notebook
def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
                                      


# In[25]:


imgs, labels = next(train_batch)


# In[26]:


plots(imgs, titles=labels)


# In[27]:


labels


# In[28]:


from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.optimizers import Adam 

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense





model = Sequential()
model.add(Convolution2D(32, (3 ,3), padding='same', input_shape=(224,224,3)))
model.add(Activation('relu'))
model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('softmax'))


# In[29]:


model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=0.01), metrics=['accuracy'])


# In[30]:


#accuracy is calculated by taking the training and comparing it with the validation.

model.fit_generator(train_batch, steps_per_epoch=30,
                    validation_data = valid_batch, validation_steps=10, nb_epoch=5, verbose=2)


# In[31]:


test_imgs, test_labels = next(test_batch)   
plots(test_imgs, titles=test_labels)


# In[32]:


test_labels


# In[33]:


test_labels = test_labels[:10,]
test_labels


# In[34]:


predictions = model.predict_generator(test_batch, steps = 1, verbose = 0)


# In[35]:


predictions = np.round(predictions, 2)


# In[36]:


predictions


# In[37]:


from sklearn.metrics import confusion_matrix



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels.argmax(axis=1), predictions.argmax(axis=1))


# In[38]:


import itertools
from itertools import chain
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[39]:


cm_plot_labels = ["versicolor", "virginica", "setosa"]     
plot_confusion_matrix(cm, cm_plot_labels, title = 'Confusion matrix')


# In[40]:


from keras.applications import VGG16
vgg16_model = keras.applications.vgg16.VGG16() 


# In[41]:


vgg16_model.summary()


# In[42]:


model = Sequential()
for layer in vgg16_model.layers:   
    model.add(layer)


# In[43]:


for layer  in model.layers:   
    layer.trainable = False


# In[44]:


model.add(Dense(3,activation = 'softmax'))


# In[45]:


model.summary()


# In[46]:


model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=0.01), metrics=['accuracy'])


# In[47]:


#accuracy is calculated by taking the training and comparing it with the validation.
model.fit_generator(train_batch, steps_per_epoch=30,
                    validation_data = valid_batch, validation_steps=10, nb_epoch=5, verbose=2)


# In[48]:


test_imgs, test_labels = next(test_batch)   
plots(test_imgs, titles=test_labels)


# In[49]:


test_labels = test_labels[:10,]
test_labels


# In[50]:


predictions = model.predict_generator(test_batch, steps = 1, verbose = 0)


# In[51]:


from sklearn.metrics import confusion_matrix



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels.argmax(axis=1), predictions.argmax(axis=1))


# In[52]:


cm_plot_labels = ["versicolor", "virginica", "setosa"]     
plot_confusion_matrix(cm, cm_plot_labels, title = 'Confusion matrix')

