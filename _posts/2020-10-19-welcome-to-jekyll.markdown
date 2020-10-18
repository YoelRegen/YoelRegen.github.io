---
layout: post
title:  "Deep Learning Malaria Cell Detection"
date:   2020-10-19 04:40:15 +0700
categories: jekyll update
---
First, you can download the dataset and information about it at [US National Library of Medicine (Index 41)][US National Library of Medicine (Index 41)].

[US National Library of Medicine (Index 41)]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7277980/

Now we need to import libraries we needed for making model.

{% highlight python %}
#For read and visualize dataset
import os
import glob
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import seaborn as sns
import numpy as np

#For image preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#For Modeling
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.applications import vgg19
from tensorflow.keras import Model

#For saving weights and visualize training process
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
import datetime

#For evaluate model
from PIL import Image
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
{% endhighlight %}


Then we need to read the data. My approach here by getting dataset that already on my Google Drive before, then unzip it. After that remove the zip file.

{% highlight python %}
zip_path = '/content/drive/My\ Drive/BCML/Project/cell_images_corrected.zip'

!cp {zip_path} /content/

!cd /content/

!unzip -q /content/cell_images_corrected.zip -d /content

!rm /content/cell_images_corrected.zip
{% endhighlight %}

Now we can see the dataset files at Colab temporary storage.

![image01](/assets/img/img_01.PNG)

Then we doing some pathing to each categorical images folders, and we need to know how many images on it.

{% highlight python %}
base_dir = os.path.join('./cell_images_corrected')
infected_dir = os.path.join(base_dir,'True/True_parasitized')
healthy_dir = os.path.join(base_dir,'True/True_uninfected')

false_infected_dir = os.path.join(base_dir,'False/False_parasitized')
false_healthy_dir = os.path.join(base_dir,'False/False_uninfected')

infected_files = glob.glob(infected_dir+'/*.png')
healthy_files = glob.glob(healthy_dir+'/*.png')

false_infected_files = glob.glob(false_infected_dir+'/*.png')
false_healthy_files = glob.glob(false_healthy_dir+'/*.png')

#Check how many images on each categories
len(infected_files), len(healthy_files), len(false_infected_files), len(false_healthy_files)
{% endhighlight %}

![image1](/assets/img/img_1.PNG)

Now we can see how many the images. The next step is visualize the data since we must know the data itself.

{% highlight python %}
print('True Parasitized Cell Sample: \n')
plt.figure(figsize = (15,15))
for i in range(5):
    plt.subplot(5, 5, i+1)
    img = cv2.imread(infected_files[i])
    plt.imshow(img)
plt.show()
{% endhighlight %}

![image2](/assets/img/img_2.PNG)

{% highlight python %}
print('True Uninfected Cell Sample: \n')
plt.figure(figsize = (15,15))
for i in range(5):
    plt.subplot(5, 5, i+1)
    img = cv2.imread(healthy_files[i])
    plt.imshow(img)
plt.show()
{% endhighlight %}

![image3](/assets/img/img_3.PNG)

{% highlight python %}
print('False Parasitized Cell Sample: \n')
plt.figure(figsize = (15,15))
for i in range(5):
    plt.subplot(5, 5, i+1)
    img = cv2.imread(false_infected_files[i])
    plt.imshow(img)
plt.show()
{% endhighlight %}

![image4](/assets/img/img_4.PNG)

{% highlight python %}
print('False Uninfected Cell Sample: \n')
plt.figure(figsize = (15,15))
for i in range(5):
    plt.subplot(5, 5, i+1)
    img = cv2.imread(false_healthy_files[i])
    plt.imshow(img)
plt.show()
{% endhighlight %}

![image5](/assets/img/img_5.PNG)

From the images, the main thing we can see clearly is purple spot on the images. The purple spot on blood cell called Merozoite caused by parasite commonly named Plasmodium that caused Malaria disease. The blood cell which had Merozoite identified as Parasitized, and blood cell not have Merozoite identified as Uninfected. However to see this blood cell sometimes require some substance to increase the visibility of Merozoid. And sometimes with this method will leave some spot on Uninfected cell but got recognized as Parasitized cell.

The next step is doing Image Preprocessing and Split Data for training and validation set. But we dont need False Infected and False Healthy data Since they are mislabeled. My approach here not using any Image Augmentation, train and validation data got splitted to 80:20. Since I'm want to use binary_crossentropy for Loss Function, I'm using 'binary' for class_mode. And resize all images to 48x48.

{% highlight python %}
dataset_dir = '/content/cell_images_corrected/'

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(os.path.join(dataset_dir, 'True'), class_mode='binary', batch_size=8, target_size=(48, 48), subset='training', shuffle=True)
val_generator = datagen.flow_from_directory(os.path.join(dataset_dir, 'True'), class_mode='binary', batch_size=8, target_size=(48, 48), subset='validation', shuffle=False)
{% endhighlight %}

![image6](/assets/img/img_6.PNG)

We need to know from two classes which one labeled as '0' and '1'

{% highlight python %}
train_generator.class_indices
{% endhighlight %}

![image7](/assets/img/img_7.PNG)

Then we make the model for training. On this approach using 4 x Convolutional layer and Max Pooling for image filtering and 1 x Fully Connected layer. And for output using 1 output neuron because we using binary_crossentropy as loss function. So the output will result binary(0,1).

{% highlight python %}
model = Sequential()

#Input layer using 16 filter with 3x3 kernel then 2x2 Max Pooling then Dropout with rate 0.2
model.add(Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(48, 48, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

#Layer using 32 filter with 3x3 kernel then 2x2 Max Pooling then Dropout with rate 0.2
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))


model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())

#Layer using 128 neuron then Dropout(0.3 rate)
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

#Using Sigmoid Activation Function for binary
model.add(Dense(1, activation='sigmoid'))

#Using Adam optimizer
opt = Adam(lr=0.0001)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
{% endhighlight %}

Then we will get summarized model output.

![image8](/assets/img/img_8.PNG)

Then we apply Checkpoint and EarlyStopping. EarlyStopping make the training stop when validation loss not getting lower for the next 3 epoch. And Checkpoint will save the best weigths based on EarlyStopping result and the epoch. Then we apply Checkpoint and EarlyStopping we created before to callbacks parameters on training.

{% highlight python %}
#Checkpoint when training
filepath="weights-best.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_loss',patience=3)

callbacks_list = [checkpoint,early_stop]
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
callbacks_list.append(TensorBoard(logdir, histogram_freq=1))

train_steps_per_epoch = train_generator.n // train_generator.batch_size
val_steps_per_epoch = val_generator.n // val_generator.batch_size

history = model.fit_generator(train_generator, steps_per_epoch=train_steps_per_epoch, epochs=75,
                              validation_data=val_generator, verbose=0, validation_steps=val_steps_per_epoch, callbacks=callbacks_list)
{% endhighlight %}

The training result achieved randomly, which means everytime we doing some training on this model will not get the same result numbers.

![image9](/assets/img/img_9.PNG)

Now let's try to print our CNN layers.

{% highlight python %}
#load the model
#summarize filter shapes
for layer in model.layers:
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# get filter weights
	filters, biases = layer.get_weights()
	print(layer.name, filters.shape)
{% endhighlight %}

![image10](/assets/img/img_10.PNG)

The result is same with our model we created before. Then we can try visualize filter on layer 0

{% highlight python %}
#Visualize Layer 0
filters, biases = model.layers[0].get_weights()
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
n_filters, ix = 6, 1
for i in range(n_filters):
	f = filters[:, :, :, i]
	for j in range(3):
		ax = pyplot.subplot(n_filters, 3, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		pyplot.imshow(f[:, :, j], cmap='gray')
		ix += 1
pyplot.show()
{% endhighlight %}

![image11](/assets/img/img_11.PNG)

For each filter we can see there is a brighter block and darker block. More brighter the block indicates higher weights the filter got from that block. Then we visualize features that got activated when training process.

{% highlight python %}
#Visualize activated feature on layer 0
img = load_img('/content/cell_images_corrected/True/True_parasitized/C100P61ThinF_IMG_20150918_144104_cell_162.png', target_size=(64, 64))
img = img_to_array(img)
img = expand_dims(img, axis=0)
img = preprocess_input(img)

inp = model.inputs
outpt = model.layers[0].output
ftr = Model(inputs=inp, outputs=outpt)

frr = ftr.predict(img)
print(frr.shape)
fig = plt.figure(figsize=(10,10))
for i in range(16):
  ax=fig.add_subplot(8,8,i+1)
  ax.imshow(frr[0,:,:,i],cmap='gray')
{% endhighlight %}

![image12](/assets/img/img_12.PNG)

White marks on each images means features that got activated by layers. We can see the cell, Merozoite, or even the background got activated. Now we visualize epoch progress by plot with Tensorboard.

{% highlight python %}
%load_ext tensorboard
%tensorboard --logdir logs
{% endhighlight %}

![image13](/assets/img/img_13.PNG)

![image14](/assets/img/img_14.PNG)

From Tensorboard we can see each train accuracy, validation accuracy, train loss and validation loss for each epoch got when training. Because we use EarlyStopping on validation loss before, epoch stopped after validation loss got no lower for the next 3 epoch, even we stated the epoch to 75. From this we can see for both accuracy and loss train and validation got little difference. Next step we will evaluate our model. First we use Classification Report.

{% highlight python %}
pred_labels = model.predict_classes(val_generator)
test_labels = val_generator.classes

print(classification_report(test_labels, pred_labels))
{% endhighlight %}

![image15](/assets/img/img_15.PNG)

From classification report, Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. Recall is the ratio of correctly predicted positive observations to the all observations in actual class - yes. Accuracy is ratio of correctly predicted positive and negatif to all data. And F1 Score is the weighted average of Precision and Recall. Usually F1 Score got more accurate and usefull than Accuracy. We can see our score is good based on them.

The next step we need to visualize Confusion Matrix to evaluate our model more accurately.

{% highlight python %}
conf = confusion_matrix(test_labels, pred_labels)
plt.figure(figsize=(5,3))
sns.set(font_scale=1.2)
ax = sns.heatmap(conf, annot=True, xticklabels=['Malaria','Healthy'], yticklabels=['Malaria','Healthy'], cbar=False, fmt='1', cmap='Reds')
plt.yticks(rotation=0)
ax.xaxis.set_ticks_position('top')
plt.show()
{% endhighlight %}

![image16](/assets/img/img_16.PNG)

From the Confusion Matrix we can see 2 labels as classification category. From left to right is how to read actual labels(y axis), and from top to bottom to read predicted labels(x axis). Which means for malaria, our models predicts 2613 images of 2626 true malaria blood cell images. And  predicts 2618 images of 2605 true healthy or uninfected blood cell images. From top left box to the right it's called True Positive, False Positive, False Negative and True Negative, as value of calculation for 4 scoring at Classification Report we used before.

The last one we need to try to predict new images which model never preprocess it before. First we need to load the model we saved before with Checkpoint. Then we compile using the same optimizer, loss function and metrics on our model before.

{% highlight python %}
#Loading model
model_filename = 'weights-best.h5'

model.load_weights(model_filename)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
{% endhighlight %}


Then let's use images from mislabeled data for prediction.

{% highlight python %}
from keras.preprocessing import image
img_path = '/content/cell_images_corrected/False/False_uninfected/C101P62ThinF_IMG_20150918_151006_cell_56.png'
img = load_img(img_path, target_size=(48,48))
plt.imshow(img)

x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)

images = np.vstack([x])
val = model.predict(images)
if val == 0:
    plt.title('Malaria')
else:
    plt.title('Healthy')
{% endhighlight %}

![image17](/assets/img/img_17.PNG)

From images folder name labeled as False Uninfected, which means data inside contain parasitized blood cell images. Which means our model predict the images correctly.

So we finished create a CNN model for predicting Malaria from blood cell images. Even the model giving good predictions, this model not guarantee 100% right predictions and CANNOT be use for medical purposes since this model created for reseach and deep learning knowledge.
