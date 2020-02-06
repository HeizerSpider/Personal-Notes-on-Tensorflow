# Tensorflow

### 1) Import dataset
### 2) Explore the data
### 3) Preprocessing of data
### 4) Building the model (Setting up of layers and compiling of model)
### 5) Training the model
### 6) Evaluation of Accuracy
### 7) Making predictions from gathered results

##### Library Setup
```
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
```

##### 1) Import dataset
```
<variable_name> = keras.datasets.<dataset_name> 
(train_images, train labels), (test_images, test_labels) = <variable_name>.load_data()
```
Training set arrays: train_images & train_labels (model uses these to 'learn')  
Model is tested against test_images and test_labels arrays  
Labels are an array with numbers corresponding to a certain class of items (eg. item/characteristic)  
Each image mapped to a certain label (and thus they are classified)  
(Images are just 2d array of numbers ranging between 0-255)

##### 2) Explore the data
-train_images.shape:   
(shows the number of images in the dataset and also how each image is represented(in terms of pixels x pixels))  
eg. (10000, 30, 30)

-train_labels:  
(represents an array of labels, corresponding to a class of objects/characteristics as mentioned)  
eg. array([0, 4, 3, 7 ... 4, 5, 9], dtype=uint8)  
dtype=uint8 (refers to array type, in this case uint8==Unsigned integer (0 to 255))

-len(<array_name>)  
(gives the length of an array)  
eg train_images has a length of 10000 as seen previously

##### 3) Preprocessing of data
All data must be preprocessed before training can happen (For supervised learning such as what we are doing now). Why?  
It improves the quality of the raw experimental data - ie eliminating experimental error/noise  
There are many forms of data preprocessing based on the different types of data: 
<https://towardsdatascience.com/image-pre-processing-c1aec0be3edf>
<https://www.sciencedirect.com/topics/engineering/data-preprocessing>

```
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
```

Scale the values to a value between range 0-1 before feeding to the neural network:
```
train_images = train_images / 255.0  
test_images=test_images / 255.0
```

Displaying and verifiation of data:
```
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
```

##### 4) Building the model (Setting up of layers and compiling of model)
Building block of a neural netowrk: layers  
```
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
```
....Flatten: transforms format of images into a 2 dimensional array (28x28 pixels), which is to unstack rows of pixels in the image and line them up (reformating of data)  
....Dense: two layers (densely/fully connected), first layer 128 nodes, second layer 10 nodes which returns an array of 10 probability scores that sum to 1

Compiling the model:
Loss function —This measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction.
Optimizer —This is how the model is updated based on the data it sees and its loss function.
Metrics —Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.
```
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

##### 5) Training the model
Feed the training data (stored in train_images & train_labels to the model)  
Models will learn to associate images and labels and make predictipons for the test set  
Loss and accuracy will be displayed
```
model.fit(train_images, train_labels, epochs=10)
```

##### 6) Evaluate accuracy
Comparing the model with a test dataset
```
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
```
If accuracy on test dataset<training dataset == overfitting (when a machine learning model perofrms worse on new, previously unseen inputs than on the training data)

##### 7) Making predictions from gathered results
```
predictions = model.predict(test_images)
predictions[0]  #checking the first prediction
np.argmax(predictions[0]) #checking which label has the highest confidence
#OR
test_labels[0]
#both will give the value of eg. 9
```

Can be graphed out


