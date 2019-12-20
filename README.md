# HighPreciseKannadaDigitRecogniser
CNN Based high precise MNIST like Kannada digit recognizer is created for a competition is organized by kaggle.
With this solution, I have completed this competition in top 4%


![Project](https://github.com/mcagriaksoy/HighPreciseKannadaDigitRecogniser/blob/master/kannada.png)

# About this dataset

Bored of MNIST?

The goal of this competition is to provide a simple extension to the classic MNIST competition we're all familiar with. Instead of using Arabic numerals, it uses a recently-released dataset of Kannada digits.

Kannada is a language spoken predominantly by people of Karnataka in southwestern India. The language has roughly 45 million native speakers and is written using the Kannada script.

This competition uses the same format as the MNIST competition in terms of how the data is structured, but it's different in that it is a synchronous re-run Kernels competition. You write your code in a Kaggle Notebook, and when you submit the results, your code is scored on both the public test set, as well as a private (unseen) test set.
Technical Information

All details of the dataset curation has been captured in the paper titled: Prabhu, Vinay Uday. "Kannada-MNIST: A new handwritten digits dataset for the Kannada language." arXiv preprint arXiv:1908.01242 (2019)

On the originally-posted dataset, the author suggests some interesting questions you may be interested in exploring. Please note, although this dataset has been released in full, the purpose of this competition is for practice, not to find the labels to submit a perfect score.

In addition to the main dataset, the author also disseminated an additional real world handwritten dataset (with 10k images), termed as the 'Dig-MNIST dataset' that can serve as an out-of-domain test dataset. It was created with the help of volunteers that were non-native users of the language, authored on a smaller sheet and scanned with different scanner settings compared to the main dataset. This 'dig-MNIST' dataset serves as a more difficult test-set (An accuracy of 76.1% was reported in the paper cited above) and achieving ~98+% accuracy on this test dataset would be rather commendable.
Acknowledgments

Kaggle thanks Vinay Prabhu for providing this interesting dataset for a Playground competition.

Image reference: https://www.researchgate.net/figure/speech-for-Kannada-numbers_fig2_313113588
# Environment

Kaggle Cloud Computing, GPU Based
Tens
# My Architecture
I have used CNN architecture to build a model.

# Accuracy improvements
One-hot encode
```
# one-hot coding
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False,categories='auto')
yy = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]
encoder.fit(yy)
# transform
train_label = train_label.reshape(-1,1)
val_label = val_label.reshape(-1,1)

train_label = encoder.transform(train_label)
val_label = encoder.transform(val_label)

print('train_label shape: %s'%str(train_label.shape))
print('val_label shape: %s'%str(val_label.shape))
```
Image transform
```
plt.imshow(train_image[13].reshape(28,28))
plt.show()
print(train_image[13].shape)

train_image = train_image/255.0
val_image = val_image/255.0
test_image = test_image/255.0

train_image = train_image.reshape(train_image.shape[0],28,28,1)
val_image = val_image.reshape(val_image.shape[0],28,28,1)
test_image = test_image.reshape(test_image.shape[0],28,28,1)
print('train_image shape: %s'%str(train_image.shape))

print('train_image shape: %s'%str(train_image.shape))
print('val_image shape: %s'%str(val_image.shape))
```
# Results 
My results are: 0.99200 accuracy (due to kaggle competition result)
![Project](https://github.com/mcagriaksoy/HighPreciseKannadaDigitRecogniser/blob/master/accuracy.PNG)

# Conclusion

# References
https://www.kaggle.com/c/Kannada-MNIST
https://www.kaggle.com/c/digit-recognizer/
https://en.wikipedia.org/wiki/Kannada
https://arxiv.org/abs/1908.01242
https://github.com/vinayprabhu/Kannada_MNIST

