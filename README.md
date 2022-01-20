# Breast Cancer Classification
## Supervised learning 
Academic project about classification using CNN.   
The main goal is to estimate if the patient has forms of cancer or not, with such context we consider it necessary to minimize false negative prediction to ensure the model is rarely wrong when it estimate there is no traces of cancer.  
We use existing unbalanced labeled data to train a Convolutional Neural Network.    



# Proceeding 
- Exploratory Data Analysis
- Pre processing
- Classification - CNN
    - Model with adjusted weights
    - Model with adjusted weights + data augmentation
    - Transfer learning

# Abstract
CNN Classifier Architecture  
![](data/models_architecture/CNN.png)

The model with adjusted weights perform well on the test set with a precision of 93% and a recall of 87% for negative predictions. (60'000 samples)  

