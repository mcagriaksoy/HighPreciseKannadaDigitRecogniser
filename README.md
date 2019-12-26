# High Accuracy Kannada Digits Recogniser

Kannada digits are diﬀerent than standard arabic digits. Over 50 million people that liver India uses those digits in their daily life. Kannada MNIST dataset is prepared like MINIST but the detection of numbers are harder than standard arabic digits. CNN Based high precise MNIST like Kannada digit recognizer is created for a competition is organized by kaggle. With this solution, The competition is completed in top 4 percentage.

![Project](https://github.com/mcagriaksoy/HighPreciseKannadaDigitRecogniser/blob/master/kannada.png)

# About this dataset

Competition Info:
The goal of this competition is to provide a simple extension to the classic MNIST competition we're all familiar with. Instead of using Arabic numerals, it uses a recently-released dataset of Kannada digits.

Kannada is a language spoken predominantly by people of Karnataka in southwestern India. The language has roughly 45 million native speakers and is written using the Kannada script.

This competition uses the same format as the MNIST competition in terms of how the data is structured, but it's different in that it is a synchronous re-run Kernels competition. You write your code in a Kaggle Notebook, and when you submit the results, your code is scored on both the public test set, as well as a private (unseen) test set.
Technical Information

All details of the dataset curation has been captured in the paper titled: Prabhu, Vinay Uday. "Kannada-MNIST: A new handwritten digits dataset for the Kannada language." arXiv preprint arXiv:1908.01242 (2019)

On the originally-posted dataset, the author suggests some interesting questions you may be interested in exploring. Please note, although this dataset has been released in full, the purpose of this competition is for practice, not to find the labels to submit a perfect score.

In addition to the main dataset, the author also disseminated an additional real world handwritten dataset (with 10k images), termed as the 'Dig-MNIST dataset' that can serve as an out-of-domain test dataset. It was created with the help of volunteers that were non-native users of the language, authored on a smaller sheet and scanned with different scanner settings compared to the main dataset. This 'dig-MNIST' dataset serves as a more difficult test-set (An accuracy of 76.1% was reported in the paper cited above) and achieving ~98+% accuracy on this test dataset would be rather commendable.
Acknowledgments

Kaggle thanks Vinay Prabhu for providing this interesting dataset for a Playground competition.

# Environment

Kaggle Cloud Computing, GPU Based
Tensorflow 2.0, Keras

# My Architecture
I have used CNN architecture to build a model. To increase accuracy, A different activation function is used. The function is named swish is not generally described on Keras. So, I need to define as custom activation function which simply Sigmoid*X .
It increases the time complexity but also increases the accuracy.

Model Summary:

I have used quite complex solution and architecture for digit recognizer. Layers can be seen on below:
```
model = Sequential()
get_custom_objects().update({'swish': Activation(swish )})
model.add(Conv2D(64, kernel_size= (3,3), input_shape=(28, 28, 1),padding='same'))

model.add(BatchNormalization(momentum=0.5, epsilon=1e-5, gamma_initializer="uniform"))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='swish'))
model.add(BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"))
model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.35))

model.add(Conv2D(128, kernel_size =(3,3),padding='same', activation='swish'))
model.add(BatchNormalization(momentum=0.2, epsilon=1e-5, gamma_initializer="uniform"))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(128,(3,3), padding='same', activation='swish' ))
model.add(BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"))
model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.35))

model.add(Conv2D(256, kernel_size = (3,3), padding='same', activation='swish'))
model.add(BatchNormalization(momentum=0.2, epsilon=1e-5, gamma_initializer="uniform"))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(256, kernel_size= (3,3) ,padding='same', activation='swish'))
model.add(BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"))
model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.35))

model.add(Flatten())
model.add(Dense(256))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))
model.summary()
```

# Swish Activation Function

The unique part of model is activation function. swish activation function is used in layers.
![Project](https://github.com/mcagriaksoy/HighPreciseKannadaDigitRecogniser/blob/master/swish.png)
# Optimizer that I used
We have many optimizer algorithms and methods in deep learning. In this project RMSProp is used. Due to research Adam behves slightly better than RMSProp but, In my solution RMSProp behaves better. On the other hand, earlystopping and ReduceLROnPlateau features of keras are used. They decrease training time and prevent overfitting.

Comparison between optimizers:
![Project](https://github.com/mcagriaksoy/HighPreciseKannadaDigitRecogniser/blob/master/gif.gif)

# Accuracy improvements

To increase dataset and obtain overfitting, I have manupilate the images and create new ones to train model. They are shifted, rotated a bit and some minor noise cancellition process happened.

# Results 
My results are: 0.99200 accuracy (due to kaggle competition result)
![Project](https://github.com/mcagriaksoy/HighPreciseKannadaDigitRecogniser/blob/master/accuracy.PNG)

# Conclusion

In conclusion, a CNN model is created for Kannada digit recognizer. The results are brilliant for me. 
The unique part of project is activation function part. (Swish is used) 
In the future, I would like to adapt GNN in this solution. Because in this kannada digits they have more feature than standart arabic numbers. So, I can use some graph networks to detect cross lines on the image.


# References
https://www.kaggle.com/c/Kannada-MNIST
https://www.kaggle.com/c/digit-recognizer/
https://en.wikipedia.org/wiki/Kannada
https://arxiv.org/abs/1908.01242
https://www.researchgate.net/figure/speech-for-Kannada-numbers_fig2_313113588
https://github.com/vinayprabhu/Kannada_MNIST

