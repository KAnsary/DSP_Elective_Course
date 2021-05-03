
# coding: utf-8

# # <b>Introduction:</b>
# It is required to design a hand-written numeral recognition system that being trained using given data. Features should be extracted from that data using different methods, train the model with different algorithms and then test the model and compare between these methods accuraces.
# 
# ## Features generation:
#  - Centroid features.
#  - AutoEncoder.
# 
# ## Classification algorithms:
#  - k-means clustering.
#  - GMM.
#  - SVM.
# 

# In[49]:


#Initializing needed libaries
import scipy as sp
import numpy as np
import pandas as pd
import matplotlib as plt
from scipy.fftpack import dct as dct
from scipy import io as spio
from scipy import ndimage as img
from random import randint
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import time
from sklearn.metrics import confusion_matrix
from scipy import linalg as la
from numpy.linalg import inv
from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as K
from scipy.ndimage.measurements import center_of_mass
import math
import warnings
warnings.filterwarnings('ignore')
get_ipython().magic(u'matplotlib inline')


# # <font color='navy' size=4><b>Core Functions:</b></font>
# ### -Extract Data
# <p>This function handles the input data, first it reads the input features and labels, then it reshapes features into an image pixels format while labeles transform it into class number instead of hot one.</p>
# 
# ### -Standarize
# <p>To make data in standard form with mean=0 and variance of 1, this function takes the input data and standarize them.</p>
# <p> The standarization process decreases the the values of input data making information dense in smaller values while keeping all information, due to smaller values computational process becomes faster which is in favour of the algorithm.</p> 
# 
# ### -Unroll
# <p>Unrolling of input into a single vector for other functions.</p>
# 
# ### -dct_2D
# <p>DCT is a powerful transformation for features which decreases number of features dramatically while keeping most of information and variations in data.</p>
# <p> first 2D DCT transformation is applied on the data then Zigzag reading of DCT coefficient to make the most of the transformation. </p>
# 
# ### -pca_fit & pca_trans
# <p>PCA is another powerful transformation where it reduces input features into smaller number of feature, it reduces the number of dimensions of the input data while keeping high variations of the data.</p>
# <p> pca_fit forms the model on the training data, while pca_trans transforms the test features into the same model of the training data.</p>
# 
# ### -KMeans
# <p>K-Means is one of the most known clusrting algorithms, in order to use it with our classification problem it was applied on each class training data in order to produce the means of the each class data, then at test time nearest class is assigned to the input vector with the nearest distance from cluster of the assigned class.</p>
# 
# ### -GMM
# <p>Guassian Mixture Model is another important algorithm of clustering, it takes longer time for training however it helps a lot in increasing accuracy when compared to kmeans using same number of clusters</p>
# 
# ### -predict_acc
# <p>In this function it estimates which class should be assigned to the test example, and then calculates the accuracy of the predictions according to the true labels.</p>
# 
# ### -accuracy 
# <p>In this function it calculates the correctly predicted classes ratio to the whole test set and returns the accuracy.</p>
# 
# ### -find_class
# <p>This function estimates the class that should be assigned to the currrebt test vector of features, this function is uded by predict_acc</p>
# 
# ### -Centroids
# <p> Slicing the input image into 9 slices that have the same size and returns an array containing these slices.</p>
# 
# ### -Get_Centroid_Features
# <p> Taking these slices that (centroids) function returns, and then compute the distance between the center of mass of each slice and center of mass of the image and finally returns these distances as the desired features.</p>
# 
# ### -features_diagonalization
# <p> We can diagonalize the covariance matrix of the new features using this function that returns a diagonal matrix containing eignvalues of the features.</p>
# 

# In[50]:


#Functions definations 
#extract data from Mat format and normalize the image
def extract(Data):
    features=[]
    output=[]
    for i in range (Data.shape[0]):
            features.append(Data[i,0])
            output.append(Data[i,1])
    features=np.array(features)
    output=np.array(output)
    features=np.reshape(features,[features.shape[0]*features.shape[1],28,28])/1.0
    output=np.reshape(output,[output.shape[0]*output.shape[1],output.shape[2]])
    output=[np.where(r==1)[0][0] for r in output]
    return features, np.array(output)
def standarize(x):
    stnd=[]
    for i in range(x.shape[0]):
        scaler = StandardScaler()
        scaler.fit(x[i]) 
        temp=scaler.transform(x[i])
        stnd.append(temp)
    return np.array(stnd)
def unroll(a):
    a=a.reshape(a.shape[0],a.shape[1]*a.shape[2])
    return a
#2D dct
def dct_2D(x):
    a=[]
    for i in range (x.shape[0]):
        x_dct=dct(dct(x[i],norm='ortho').T,norm='ortho').T;
        a.append([x_dct[0,0], x_dct[0,1], x_dct[1,0], x_dct[2,0], x_dct[1,1],                  x_dct[0,2], x_dct[0,3], x_dct[1,2], x_dct[2,1], x_dct[3,0],                  x_dct[4,0], x_dct[3,1], x_dct[2,2], x_dct[1,3], x_dct[0,4],                  x_dct[0,5], x_dct[1,4], x_dct[2,3], x_dct[3,2], x_dct[4,1]])   
    return np.array(a)
#PCA
def pca_fit(x,n):
    #x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=n)
    pca.fit(x)
    pca_comp=pca.transform(x)
    var=sum(pca.explained_variance_ratio_)*100
    return pca_comp,var,pca
def pca_trans(x,pca):
    #x = StandardScaler().fit_transform(x)
    pca_comp=pca.transform(x)
    var=sum(pca.explained_variance_ratio_)*100.0
    return pca_comp,var
#K-Means
def kmeans(clusters,classes_n,Features_Train,class_margin):
    kmeans_=[]
    for i in range (classes_n):  
        kmeans_temp=KMeans(n_clusters=clusters,n_init=10,max_iter=5000,algorithm='full',random_state=0).        fit(Features_Train[i*class_margin:i*class_margin+class_margin-1])
        kmeans_.append(kmeans_temp.cluster_centers_)
    kmeans_=np.array(kmeans_)
    return kmeans_
#GMM 
def GMM(Mixtures,classes_n,Features_Train,class_margin):
    G=[]
    for i in range (classes_n):  
        G_temp=GaussianMixture(n_components=Mixtures,n_init=10,max_iter=5000,covariance_type='full',random_state=0).        fit(Features_Train[i*class_margin:i*class_margin+class_margin-1])
        G.append(G_temp.means_)
    G=np.array(G)
    return G
#Predict
def predict_acc(test_features,label_set,model):
    Y_predict=np.zeros_like(label_set)
    for i in range (Y_predict.shape[0]):
        Y_predict[i]=find_class(test_features[i],model)
    acc1=accuracy(label_set,Y_predict)
    return acc1,Y_predict
#accuracy calc 
def accuracy(original,predicted):
    acc=original-predicted
    acc[acc != 0] = 1
    acc=(np.count_nonzero(acc == 0)*1.0/original.shape[0])*100.0
    return acc
#class decision
def find_class(x,y):
    min_d=np.ones(y.shape[0])*100000000.0
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            temp=np.linalg.norm(x-y[i][j])
            if temp<min_d[i]:
                min_d[i]=temp
    min_class_idx=np.argmin(min_d)
    return min_class_idx
#Grid slicing
def Centroids(x):
    Slice_1=[]
    Slice_2=[]
    Slice_3=[]
    Slice_4=[]
    Slice_5=[]
    Slice_6=[]
    Slice_7=[]
    Slice_8=[]
    
    for i in range (x.shape[0]):
        Slice_1.append(np.hstack(x[i][0:9,0:9]))
        Slice_2.append(np.hstack(x[i][0:9,9:18]))
        Slice_3.append(np.hstack(x[i][0:9,18:27]))
        
        Slice_4.append(np.hstack(x[i][9:18,0:9]))
        Slice_5.append(np.hstack(x[i][9:18,18:27]))
        
        Slice_6.append(np.hstack(x[i][18:27,0:9]))
        Slice_7.append(np.hstack(x[i][18:27,9:18]))
        Slice_8.append(np.hstack(x[i][18:27,18:27]))
    
    Slices = { "S1": np.array(Slice_1).reshape(x.shape[0],9,9),
               "S2": np.array(Slice_2).reshape(x.shape[0],9,9),
               "S3": np.array(Slice_3).reshape(x.shape[0],9,9),
               "S4": np.array(Slice_4).reshape(x.shape[0],9,9),
               "S5": np.array(Slice_5).reshape(x.shape[0],9,9),
               "S6": np.array(Slice_6).reshape(x.shape[0],9,9),
               "S7": np.array(Slice_7).reshape(x.shape[0],9,9),
               "S8": np.array(Slice_8).reshape(x.shape[0],9,9)
             }
    return Slices
def Get_Centroid_Features(x_slices, x_train):
    Features=[]
    for i in range (x_train.shape[0]):
        stack=[]
        for j in range (1,9):
            a=np.array(center_of_mass(x_slices["S"+str(j)][i])).reshape(1,2)
            b=np.array(center_of_mass(x_train[i])).reshape(1,2)
            
            if math.isnan(a[0,0]):
                a=np.zeros((1,2))
            
            stack.append(np.linalg.norm(a-b))
        temp1=np.array(stack)
        temp2=temp1.reshape(1,8)
        Features.append(temp2)
    return np.array(Features).reshape((x_train.shape[0],8))
#feature diagnolization
def features_diagonalization(x):
    m = x.shape[1]
    covariance_matrix = (1/m) * np.dot(np.transpose(x),x)
    # covariance_matrix.shape
    e_vals, e_vecs = la.eig(covariance_matrix)
    diagonal_eignvalues = np.dot(np.dot(inv(e_vecs),covariance_matrix),e_vecs)
    return diagonal_eignvalues


# ## <font color='navy'><b>Step 1:</b></font>
# * Read Data
# * Extract the features and labels
# * Standarize the features
# * Unroll the Features
# * Show an example of handwritten image

# In[69]:


#Read Data
R_MNIST=spio.loadmat('./ReducedMNIST.mat')
R_MINST_Train=R_MNIST['SmallTrainData']
R_MINST_Test=R_MNIST['SmallTestData']
#extract features and labels
X_Train,Y_Train= extract(R_MINST_Train)
X_Train_std=standarize(X_Train)
X_Test,Y_Test= extract(R_MINST_Test)
X_Test_std=standarize(X_Test)
#unroll images
X_Train_unroll=unroll(X_Train_std)
X_Train_unroll_norm=unroll(X_Train)/255.0
X_Test_unroll=unroll(X_Test_std)
X_Test_unroll_norm=unroll(X_Test)/255.0
#show a random picture example
img_num=randint(0,X_Train.shape[0])
plt.pyplot.imshow(img.rotate(X_Train[img_num],90),origin='lower')
plt.pyplot.gray()
plt.pyplot.show()
print("Label = "+ str(Y_Train[img_num])+" image = "+ str(img_num))


# ## <font color='navy'><b>Step 2:</b></font>
# Extracting the centroid features from training and test data by calling (Centroids) and (Get_Centroid_Features) functions.

# In[52]:


#extracting centeroids features for Train and Test Data
Slices_Train=Centroids(X_Train)
Centroid_Features_Train= Get_Centroid_Features(Slices_Train,X_Train)
Slices_Test=Centroids(X_Test)
Centroid_Features_Test= Get_Centroid_Features(Slices_Test, X_Test)


# In[53]:


#plot a slice example
img_num=randint(0,X_Train.shape[0])
plt.pyplot.imshow(img.rotate(Slices_Train["S2"][img_num],90),origin='lower')
plt.pyplot.gray()
plt.pyplot.show()


# ## <font color='navy'><b>Step 3:</b></font>
# Creating Autoencoder model that extracts 10 features and training it using the given data.
# 

# ## Autoencoding:
# "Autoencoding" is a data compression algorithm where the compression and decompression functions are 1) data-specific, 2) lossy, and 3) learned automatically from examples rather than engineered by a human. Additionally, in almost all contexts where the term "autoencoder" is used, the compression and decompression functions are implemented with neural networks.
# 
# To build an autoencoder, you need three things: an encoding function, a decoding function, and a distance function between the amount of information loss between the compressed representation of your data and the decompressed representation (i.e. a "loss" function). The encoder and decoder will be chosen to be parametric functions (typically neural networks), and to be differentiable with respect to the distance function, so the parameters of the encoding/decoding functions can be optimize to minimize the reconstruction loss, using Stochastic Gradient Descent.
# 
# One reason why they have attracted so much research and attention is because they have long been thought to be a potential avenue for solving the problem of unsupervised learning, i.e. the learning of useful representations without the need for labels. Then again, autoencoders are not a true unsupervised learning technique (which would imply a different learning process altogether), they are a self-supervised technique, a specific instance of supervised learning where the targets are generated from the input data. In order to get self-supervised models to learn interesting features, you have to come up with an interesting synthetic target and loss function, and that's where problems arise: merely learning to reconstruct your input in minute detail might not be the right choice here. At this point there is significant evidence that focusing on the reconstruction of a picture at the pixel level, for instance, is not conductive to learning interesting, abstract features of the kind that label-supervized learning induces (where targets are fairly abstract concepts "invented" by humans such as "dog", "car"...). In fact, one may argue that the best features in this regard are those that are the worst at exact input reconstruction while achieving high performance on the main task that you are interested in (classification, localization, etc).

# In[55]:


encoding_dim = 10  #size of output of encoder 
Compression_ratio=784/encoding_dim
input_img = Input(shape=(784,)) #input placeholder
encoded = Dense(encoding_dim, activation='relu')(input_img) #encoding layer output
decoded = Dense(784, activation='sigmoid')(encoded) #decoding layer output
autoencoder = Model(input_img, decoded) #autoencoder model
encoder = Model(input_img, encoded) #encoder model
encoded_input = Input(shape=(encoding_dim,)) #input placeholder for decoder input
decoder_layer = autoencoder.layers[-1] # retrieve the last layer of the autoencoder model
decoder = Model(encoded_input, decoder_layer(encoded_input)) #decoder model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')#compile autoencoder mode
print(round(Compression_ratio,3))
print("Compression ratio =" , round(Compression_ratio,4))


# In[56]:


#Train the Autoencoder 
autoencoder.fit(X_Train_unroll_norm, X_Train_unroll_norm,
                epochs=70,
                batch_size=256,
                shuffle=True,
                validation_data=(X_Test_unroll_norm, X_Test_unroll_norm))


# In[57]:


#encode and decode Test Images
encoded_imgs = encoder.predict(X_Test_unroll_norm)
decoded_imgs = decoder.predict(encoded_imgs)


# In[11]:


#display Original test images and decoded Images examples
n = 10  # how many digits we will display
plt.pyplot.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.pyplot.subplot(2, n, i + 1)
    img_num=randint(0,X_Test.shape[0])
    plt.pyplot.imshow(img.rotate(X_Test[img_num].reshape(28, 28),90),origin='lower')
    plt.pyplot.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display reconstruction
    ax = plt.pyplot.subplot(2, n, i + 1 + n)
    plt.pyplot.imshow(img.rotate(decoded_imgs[img_num].reshape(28, 28),90),origin='lower')
    plt.pyplot.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.pyplot.show()


# In[58]:


#Create Encoded data for train and test datasets
encoded_Train = encoder.predict(X_Train_unroll_norm)
encoded_Test = encoder.predict(X_Test_unroll_norm)


# ## <font color='navy'><b>Step 4:</b></font>
# Using output features from the Autoencoder (10 features), train the model using K-means clustering algorithm with different number of clusters and comparing output accuracy when classifing test images.

# In[13]:


#KMeans 1 cluster AutoEncoder
tic = time.time()
kmeans_encode1=kmeans(1,10,encoded_Train,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# In[14]:


#Encoder KMeans 1 cluster Prediction 
tic = time.time()
acc_kmeans_encoded1,Y_encoded_KMeans1=predict_acc(encoded_Test,Y_Test,kmeans_encode1)
toc = time.time()
print("accuracy =",round(acc_kmeans_encoded1,2),"%") 
print("elapsed time =",round(toc-tic,4),"sec\n")
print("Confusion Matrix:")
confusion_encoded_KMeans1 = pd.crosstab(Y_Test, Y_encoded_KMeans1,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_encoded_KMeans1)


# In[15]:


#KMeans 2 cluster AutoEncoder
tic = time.time()
kmeans_encode2=kmeans(2,10,encoded_Train,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# In[16]:


#Encoder KMeans 2 cluster Prediction 
tic = time.time()
acc_kmeans_encoded2,Y_encoded_KMeans2=predict_acc(encoded_Test,Y_Test,kmeans_encode2)
toc = time.time()
print("accuracy =",round(acc_kmeans_encoded2,2),"%") 
print("elapsed time =",round(toc-tic,4),"sec\n")
print("Confusion Matrix:")
confusion_encoded_KMeans2 = pd.crosstab(Y_Test, Y_encoded_KMeans2,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_encoded_KMeans2)


# In[60]:


#KMeans 4 cluster AutoEncoder
tic = time.time()
kmeans_encode4=kmeans(4,10,encoded_Train,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# In[64]:


#Encoder KMeans 4 cluster Prediction 
tic = time.time()
acc_kmeans_encoded4,Y_encoded_KMeans4=predict_acc(encoded_Test,Y_Test,kmeans_encode4)
toc = time.time()
print("accuracy =",round(acc_kmeans_encoded4,2),"%") 
print("elapsed time =",round(toc-tic,4),"sec\n")
for t in range(5):
    print("\n")
print("Confusion Matrix:")
confusion_encoded_KMeans4 = pd.crosstab(Y_Test, Y_encoded_KMeans4,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_encoded_KMeans4)


# In[65]:


#KMeans 8 cluster AutoEncoder
tic = time.time()
kmeans_encode8=kmeans(8,10,encoded_Train,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# In[66]:


#Encoder KMeans 8 cluster Prediction 
tic = time.time()
acc_kmeans_encoded8,Y_encoded_KMeans8=predict_acc(encoded_Test,Y_Test,kmeans_encode8)
toc = time.time()
print("accuracy =",round(acc_kmeans_encoded8,2),"%") 
print("elapsed time =",round(toc-tic,4),"sec\n")
for t in range(7):
    print("\n")
print("Confusion Matrix:")
confusion_encoded_KMeans8 = pd.crosstab(Y_Test, Y_encoded_KMeans8,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_encoded_KMeans8)


# In[21]:


#KMeans 16 cluster AutoEncoder
tic = time.time()
kmeans_encode16=kmeans(16,10,encoded_Train,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# In[22]:


#Encoder KMeans 16 cluster Prediction 
tic = time.time()
acc_kmeans_encoded16,Y_encoded_KMeans16=predict_acc(encoded_Test,Y_Test,kmeans_encode16)
toc = time.time()
print("accuracy =",round(acc_kmeans_encoded16,2),"%") 
print("elapsed time =",round(toc-tic,4),"sec\n")
print("Confusion Matrix:")
confusion_encoded_KMeans16 = pd.crosstab(Y_Test, Y_encoded_KMeans16,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_encoded_KMeans16)


# ## <font color='navy'><b>Step 5:</b></font>
# Using the same features from autoencoder, but with GMM algorithm with different number of GMM.

# In[23]:


#encoded GMM 1
tic = time.time()
G_encoded1=GMM(1,10,encoded_Train,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# In[24]:


#encoded GMM 1 Predictions
tic = time.time()
acc_GMM_encoded1,Y_encoded_GMM1=predict_acc(encoded_Test,Y_Test,G_encoded1)
toc = time.time()
print("accuracy =",round(acc_GMM_encoded1,2),"%") 
print("elapsed time =",round(toc-tic,4),"sec")
print("\nConfusion Matrix:")
confusion_encoded_GMM1 = pd.crosstab(Y_Test, Y_encoded_GMM1,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_encoded_GMM1)


# In[67]:


#encoded GMM 2
tic = time.time()
G_encoded2=GMM(2,10,encoded_Train,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# In[68]:


#encoded GMM 2 Predictions
tic = time.time()
acc_GMM_encoded2,Y_encoded_GMM2=predict_acc(encoded_Test,Y_Test,G_encoded2)
toc = time.time()
print("accuracy =",round(acc_GMM_encoded2,2),"%") 
print("elapsed time =",round(toc-tic,4),"sec")
for t in range(9):
    print("\n")
print("\nConfusion Matrix:")
confusion_encoded_GMM2 = pd.crosstab(Y_Test, Y_encoded_GMM2,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_encoded_GMM2)


# In[27]:


#encoded GMM 4
tic = time.time()
G_encoded4=GMM(4,10,encoded_Train,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# In[28]:


#encoded GMM 4 Predictions
tic = time.time()
acc_GMM_encoded4,Y_encoded_GMM4=predict_acc(encoded_Test,Y_Test,G_encoded4)
toc = time.time()
print("accuracy =",round(acc_GMM_encoded4,2),"%") 
print("elapsed time =",round(toc-tic,4),"sec")
print("\nConfusion Matrix:")
confusion_encoded_GMM4 = pd.crosstab(Y_Test, Y_encoded_GMM4,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_encoded_GMM4)


# ## <font color='navy'><b>Step 6:</b></font>
# Same autoencoder 10 features with SVM algorithm with linear and nonlinear kernals.

# In[70]:


#encoded Linear SVM
tic = time.time()
svm_encoded_lin = svm.SVC(kernel='linear')
svm_encoded_lin.fit(encoded_Train,Y_Train)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec") 


# In[71]:


#encoded Linear SVM Prediction
tic = time.time()
acc_encoded_svm=accuracy_score(Y_Test,svm_encoded_lin.predict(encoded_Test))*100
toc = time.time()
print('accuracy = ',round(acc_encoded_svm,2),"%")
print("elapsed time =",round(toc-tic,4),"sec") 
for t in range(4):
    print("\n")
print("\nConfusion Matrix:")
confusion_svm_encoded_lin = pd.crosstab(Y_Test, svm_encoded_lin.predict(encoded_Test),                                    rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_svm_encoded_lin)


# In[72]:


#encoded non-Linear SVM
tic = time.time()
svm_encoded_nlin = svm.SVC(kernel='rbf')
svm_encoded_nlin.fit(encoded_Train,Y_Train)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec") 


# In[73]:


#encoded non-Linear SVM Prediction
tic = time.time()
acc_encoded_nsvm=accuracy_score(Y_Test,svm_encoded_nlin.predict(encoded_Test))*100
toc = time.time()
print('accuracy = ',round(acc_encoded_nsvm,2),"%")
print("elapsed time =",round(toc-tic,4),"sec") 
for t in range(6):
    print("\n")
print("\nConfusion Matrix:")
confusion_svm_encoded_nlin = pd.crosstab(Y_Test,svm_encoded_nlin.predict(encoded_Test),                                    rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_svm_encoded_nlin)


# ## <font color='navy'><b>Step 7:</b></font>
# Creating autoencoder model that gives 20 features instead of 10 and training it using training data.

# In[74]:


K.clear_session()#clear old model


# In[75]:


encoding_dim = 20  #size of output of encoder 
Compression_ratio=784/encoding_dim
input_img = Input(shape=(784,)) #input placeholder
encoded = Dense(encoding_dim, activation='relu')(input_img) #encoding layer output
decoded = Dense(784, activation='sigmoid')(encoded) #decoding layer output
autoencoder = Model(input_img, decoded) #autoencoder model
encoder = Model(input_img, encoded) #encoder model
encoded_input = Input(shape=(encoding_dim,)) #input placeholder for decoder input
decoder_layer = autoencoder.layers[-1] # retrieve the last layer of the autoencoder model
decoder = Model(encoded_input, decoder_layer(encoded_input)) #decoder model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')#compile autoencoder mode
print("Compression ratio =" , round(Compression_ratio,3))


# In[76]:


# Train the Autoencoder 
autoencoder.fit(X_Train_unroll_norm, X_Train_unroll_norm,
                epochs=70,
                batch_size=256,
                shuffle=True,
                validation_data=(X_Test_unroll_norm, X_Test_unroll_norm))


# In[77]:


#encode and decode Test Images
encoded_imgs = encoder.predict(X_Test_unroll_norm)
decoded_imgs = decoder.predict(encoded_imgs)


# In[37]:


#display Original test images and decoded Images examples
n = 10  # how many digits we will display
plt.pyplot.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.pyplot.subplot(2, n, i + 1)
    img_num=randint(0,X_Test.shape[0])
    plt.pyplot.imshow(img.rotate(X_Test[img_num].reshape(28, 28),90),origin='lower')
    plt.pyplot.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display reconstruction
    ax = plt.pyplot.subplot(2, n, i + 1 + n)
    plt.pyplot.imshow(img.rotate(decoded_imgs[img_num].reshape(28, 28),90),origin='lower')
    plt.pyplot.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.pyplot.show()


# In[20]:


#Create Encoded data for train and test datasets
encoded_Train = encoder.predict(X_Train_unroll_norm)
encoded_Test = encoder.predict(X_Test_unroll_norm)


# ## <font color='navy'><b>Step 8:</b></font>
# Autoencoder 20 features - K-means clustering algorithm

# In[39]:


#KMeans 1 cluster AutoEncoder
tic = time.time()
kmeans_encode1=kmeans(1,10,encoded_Train,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# In[40]:


#Encoder KMeans 1 cluster Prediction 
tic = time.time()
acc_kmeans_encoded1,Y_encoded_KMeans1=predict_acc(encoded_Test,Y_Test,kmeans_encode1)
toc = time.time()
print("accuracy =",round(acc_kmeans_encoded1,2),"%") 
print("elapsed time =",round(toc-tic,4),"sec\n")
print("Confusion Matrix:")
confusion_encoded_KMeans1 = pd.crosstab(Y_Test, Y_encoded_KMeans1,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_encoded_KMeans1)


# In[41]:


#KMeans 2 cluster AutoEncoder
tic = time.time()
kmeans_encode2=kmeans(2,10,encoded_Train,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# In[42]:


#Encoder KMeans 2 cluster Prediction 
tic = time.time()
acc_kmeans_encoded2,Y_encoded_KMeans2=predict_acc(encoded_Test,Y_Test,kmeans_encode2)
toc = time.time()
print("accuracy =",round(acc_kmeans_encoded2,2),"%") 
print("elapsed time =",round(toc-tic,4),"sec\n")
print("Confusion Matrix:")
confusion_encoded_KMeans2 = pd.crosstab(Y_Test, Y_encoded_KMeans2,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_encoded_KMeans2)


# In[43]:


#KMeans 4 cluster AutoEncoder
tic = time.time()
kmeans_encode4=kmeans(4,10,encoded_Train,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# In[44]:


#Encoder KMeans 4 cluster Prediction 
tic = time.time()
acc_kmeans_encoded4,Y_encoded_KMeans4=predict_acc(encoded_Test,Y_Test,kmeans_encode4)
toc = time.time()
print("accuracy =",round(acc_kmeans_encoded4,2),"%") 
print("elapsed time =",round(toc-tic,4),"sec\n")
print("Confusion Matrix:")
confusion_encoded_KMeans4 = pd.crosstab(Y_Test, Y_encoded_KMeans4,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_encoded_KMeans4)


# In[78]:


#KMeans 8 cluster AutoEncoder
tic = time.time()
kmeans_encode8=kmeans(8,10,encoded_Train,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# In[79]:


#Encoder KMeans 8 cluster Prediction 
tic = time.time()
acc_kmeans_encoded8,Y_encoded_KMeans8=predict_acc(encoded_Test,Y_Test,kmeans_encode8)
toc = time.time()
print("accuracy =",round(acc_kmeans_encoded8,2),"%") 
print("elapsed time =",round(toc-tic,4),"sec\n")
for t in range(3):
    print("\n")
print("Confusion Matrix:")
confusion_encoded_KMeans8 = pd.crosstab(Y_Test, Y_encoded_KMeans8,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_encoded_KMeans8)


# In[80]:


#KMeans 16 cluster AutoEncoder
tic = time.time()
kmeans_encode16=kmeans(16,10,encoded_Train,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# In[81]:


#Encoder KMeans 16 cluster Prediction 
tic = time.time()
acc_kmeans_encoded16,Y_encoded_KMeans16=predict_acc(encoded_Test,Y_Test,kmeans_encode16)
toc = time.time()
print("accuracy =",round(acc_kmeans_encoded16,2),"%") 
print("elapsed time =",round(toc-tic,4),"sec\n")
for t in range(9):
    print("\n")
print("Confusion Matrix:")
confusion_encoded_KMeans16 = pd.crosstab(Y_Test, Y_encoded_KMeans16,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_encoded_KMeans16)


# ## <font color='navy'><b>Step 9:</b></font>
# Autoencoder 20 features - GMM algorithm

# In[49]:


#encoded GMM 1
tic = time.time()
G_encoded1=GMM(1,10,encoded_Train,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# In[50]:


#encoded GMM 1 Predictions
tic = time.time()
acc_GMM_encoded1,Y_encoded_GMM1=predict_acc(encoded_Test,Y_Test,G_encoded1)
toc = time.time()
print("accuracy =",round(acc_GMM_encoded1,2),"%") 
print("elapsed time =",round(toc-tic,4),"sec")
print("\nConfusion Matrix:")
confusion_encoded_GMM1 = pd.crosstab(Y_Test, Y_encoded_GMM1,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_encoded_GMM1)


# In[51]:


#encoded GMM 2
tic = time.time()
G_encoded2=GMM(2,10,encoded_Train,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# In[52]:


#encoded GMM 2 Predictions
tic = time.time()
acc_GMM_encoded2,Y_encoded_GMM2=predict_acc(encoded_Test,Y_Test,G_encoded2)
toc = time.time()
print("accuracy =",round(acc_GMM_encoded2,2),"%") 
print("elapsed time =",round(toc-tic,4),"sec")
print("\nConfusion Matrix:")
confusion_encoded_GMM2 = pd.crosstab(Y_Test, Y_encoded_GMM2,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_encoded_GMM2)


# In[82]:


#encoded GMM 4
tic = time.time()
G_encoded4=GMM(4,10,encoded_Train,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# In[83]:


#encoded GMM 4 Predictions
tic = time.time()
acc_GMM_encoded4,Y_encoded_GMM4=predict_acc(encoded_Test,Y_Test,G_encoded4)
toc = time.time()
print("accuracy =",round(acc_GMM_encoded4,2),"%") 
print("elapsed time =",round(toc-tic,4),"sec")
for t in range(10):
    print("\n")
print("\nConfusion Matrix:")
confusion_encoded_GMM4 = pd.crosstab(Y_Test, Y_encoded_GMM4,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_encoded_GMM4)


# ## <font color='navy'><b>Step 10:</b></font>
# Autoencoder 20 features - SVM algorithm

# In[55]:


#encoded Linear SVM
tic = time.time()
svm_encoded_lin = svm.SVC(kernel='linear')
svm_encoded_lin.fit(encoded_Train,Y_Train)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec") 


# In[56]:


#encoded Linear SVM Prediction
tic = time.time()
acc_encoded_svm=accuracy_score(Y_Test,svm_encoded_lin.predict(encoded_Test))*100
toc = time.time()
print('accuracy = ',round(acc_encoded_svm,2),"%")
print("elapsed time =",round(toc-tic,4),"sec") 
print("\nConfusion Matrix:")
confusion_svm_encoded_lin = pd.crosstab(Y_Test, svm_encoded_lin.predict(encoded_Test),                                    rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_svm_encoded_lin)


# In[21]:


#encoded non-Linear SVM
tic = time.time()
svm_encoded_nlin = svm.SVC(kernel='rbf')
svm_encoded_nlin.fit(encoded_Train,Y_Train)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec") 


# In[22]:


#encoded non-Linear SVM Prediction
tic = time.time()
acc_encoded_nsvm=accuracy_score(Y_Test,svm_encoded_nlin.predict(encoded_Test))*100
toc = time.time()
print('accuracy = ',round(acc_encoded_nsvm,2),"%")
print("elapsed time =",round(toc-tic,4),"sec") 
print("\nConfusion Matrix:")
confusion_svm_encoded_nlin = pd.crosstab(Y_Test,svm_encoded_nlin.predict(encoded_Test),                                    rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_svm_encoded_nlin)


# ## <font color='navy'><b>Step 11:</b></font>
# centroid features - KMeans clustering classifing algorithm

# In[84]:


#KMeans 1 cluster centroid features
tic = time.time()
kmeans_centroid1=kmeans(1,10,Centroid_Features_Train,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# In[85]:


#centroid features KMeans 1 cluster Prediction 
tic = time.time()
acc_kmeans_centroid1,Y_centroid_KMeans1=predict_acc(Centroid_Features_Test,Y_Test,kmeans_centroid1)
toc = time.time()
print("accuracy =",round(acc_kmeans_centroid1,2),"%") 
print("elapsed time =",round(toc-tic,4),"sec\n")
for t in range(2):
    print("\n")
print("Confusion Matrix:")
confusion_centroid_KMeans1 = pd.crosstab(Y_Test, Y_centroid_KMeans1,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_centroid_KMeans1)


# In[86]:


#KMeans 2 cluster centroid features
tic = time.time()
kmeans_centroid2=kmeans(2,10,Centroid_Features_Train,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# In[87]:


#centroid features KMeans 2 cluster Prediction 
tic = time.time()
acc_kmeans_centroid2,Y_centroid_KMeans2=predict_acc(Centroid_Features_Test,Y_Test,kmeans_centroid2)
toc = time.time()
print("accuracy =",round(acc_kmeans_centroid2,2),"%") 
print("elapsed time =",round(toc-tic,4),"sec\n")
for t in range(9):
    print("\n")
print("Confusion Matrix:")
confusion_centroid_KMeans2 = pd.crosstab(Y_Test, Y_centroid_KMeans2,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_centroid_KMeans2)


# In[63]:


#KMeans 4 cluster centroid features
tic = time.time()
kmeans_centroid4=kmeans(4,10,Centroid_Features_Train,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# In[64]:


#centroid features KMeans 4 cluster Prediction 
tic = time.time()
acc_kmeans_centroid4,Y_centroid_KMeans4=predict_acc(Centroid_Features_Test,Y_Test,kmeans_centroid4)
toc = time.time()
print("accuracy =",round(acc_kmeans_centroid4,2),"%") 
print("elapsed time =",round(toc-tic,4),"sec\n")
print("Confusion Matrix:")
confusion_centroid_KMeans4 = pd.crosstab(Y_Test, Y_centroid_KMeans4,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_centroid_KMeans4)


# In[88]:


#KMeans 8 cluster centroid features
tic = time.time()
kmeans_centroid8=kmeans(8,10,Centroid_Features_Train,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# In[89]:


#centroid features KMeans 8 cluster Prediction 
tic = time.time()
acc_kmeans_centroid8,Y_centroid_KMeans8=predict_acc(Centroid_Features_Test,Y_Test,kmeans_centroid8)
toc = time.time()
print("accuracy =",round(acc_kmeans_centroid8,2),"%") 
print("elapsed time =",round(toc-tic,4),"sec\n")
for t in range(2):
    print("\n")
print("Confusion Matrix:")
confusion_centroid_KMeans8 = pd.crosstab(Y_Test, Y_centroid_KMeans8,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_centroid_KMeans8)


# In[90]:


#KMeans 16 cluster centroid features
tic = time.time()
kmeans_centroid16=kmeans(16,10,Centroid_Features_Train,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# In[91]:


#centroid features KMeans 16 cluster Prediction 
tic = time.time()
acc_kmeans_centroid16,Y_centroid_KMeans16=predict_acc(Centroid_Features_Test,Y_Test,kmeans_centroid16)
toc = time.time()
print("accuracy =",round(acc_kmeans_centroid16,2),"%") 
print("elapsed time =",round(toc-tic,4),"sec\n")
for t in range(7):
    print("\n")
print("Confusion Matrix:")
confusion_centroid_KMeans16 = pd.crosstab(Y_Test, Y_centroid_KMeans16,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_centroid_KMeans16)


# ## <font color='navy'><b>Step 12:</b></font>
# centroid features - GMM classifing algorithm

# In[92]:


#centroid features GMM 1
tic = time.time()
G_centroid1=GMM(1,10,Centroid_Features_Train,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# In[94]:


#Centroid Features GMM 1 Predictions
tic = time.time()
acc_GMM_centroid1,Y_centroid_GMM1=predict_acc(Centroid_Features_Test,Y_Test,G_centroid1)
toc = time.time()
print("accuracy =",round(acc_GMM_centroid1,2),"%") 
print("elapsed time =",round(toc-tic,4),"sec")
print("\nConfusion Matrix:")
confusion_centroid_GMM1 = pd.crosstab(Y_Test, Y_centroid_GMM1,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_centroid_GMM1)


# In[95]:


#centroid features GMM 2
tic = time.time()
G_centroid2=GMM(2,10,Centroid_Features_Train,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# In[96]:


#Centroid Features GMM 2 Predictions
tic = time.time()
acc_GMM_centroid2,Y_centroid_GMM2=predict_acc(Centroid_Features_Test,Y_Test,G_centroid2)
toc = time.time()
print("accuracy =",round(acc_GMM_centroid2,2),"%") 
print("elapsed time =",round(toc-tic,4),"sec")
for t in range(3):
    print("\n")
print("\nConfusion Matrix:")
confusion_centroid_GMM2 = pd.crosstab(Y_Test, Y_centroid_GMM2,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_centroid_GMM2)


# In[97]:


#centroid features GMM 4
tic = time.time()
G_centroid4=GMM(4,10,Centroid_Features_Train,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# In[98]:


#Centroid Features GMM 4 Predictions
tic = time.time()
acc_GMM_centroid4,Y_centroid_GMM4=predict_acc(Centroid_Features_Test,Y_Test,G_centroid4)
toc = time.time()
print("accuracy =",round(acc_GMM_centroid4,2),"%") 
print("elapsed time =",round(toc-tic,4),"sec")
for t in range(8):
    print("\n")
print("\nConfusion Matrix:")
confusion_centroid_GMM4 = pd.crosstab(Y_Test, Y_centroid_GMM4,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_centroid_GMM4)


# ## <font color='navy'><b>Step 13:</b></font>
# centroid features - SVM classifing algorithm

# In[75]:


#centroid feature Linear SVM
tic = time.time()
svm_centroid_lin = svm.SVC(kernel='linear')
svm_centroid_lin.fit(Centroid_Features_Train,Y_Train)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec") 


# In[76]:


#centroid feature Linear SVM Prediction
tic = time.time()
acc_centroid_svm=accuracy_score(Y_Test,svm_centroid_lin.predict(Centroid_Features_Test))*100
toc = time.time()
print('accuracy = ',round(acc_centroid_svm,2),"%")
print("elapsed time =",round(toc-tic,4),"sec") 
print("\nConfusion Matrix:")
confusion_svm_centroid_lin = pd.crosstab(Y_Test, svm_centroid_lin.predict(Centroid_Features_Test),                                    rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_svm_centroid_lin)


# In[77]:


#centroid feature non-Linear SVM
tic = time.time()
svm_centroid_nlin = svm.SVC(kernel='rbf')
svm_centroid_nlin.fit(Centroid_Features_Train,Y_Train)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec") 


# In[78]:


#centroid non-Linear SVM Prediction
tic = time.time()
acc_centroid_nsvm=accuracy_score(Y_Test,svm_centroid_nlin.predict(Centroid_Features_Test))*100
toc = time.time()
print('accuracy = ',round(acc_centroid_nsvm,2),"%")
print("elapsed time =",round(toc-tic,4),"sec") 
print("\nConfusion Matrix:")
confusion_svm_centroid_nlin = pd.crosstab(Y_Test,svm_centroid_nlin.predict(Centroid_Features_Test),                                    rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_svm_centroid_nlin)


# In[99]:


X_Train_conc=np.concatenate((encoded_Train,Centroid_Features_Train),axis=1)
X_Test_conc=np.concatenate((encoded_Test,Centroid_Features_Test),axis=1)


# In[102]:


#concatenated K-Means
tic = time.time()
kmeans_conc=kmeans(16,10,X_Train_conc,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# In[104]:


#concatenated kmeans Predictions
tic = time.time()
acc_kmeans_conc,Y_conc=predict_acc(X_Test_conc,Y_Test,kmeans_conc)
toc = time.time()
print("accuracy =",round(acc_kmeans_conc,2),"%")
print("elapsed time =",round(toc-tic,4),"sec")
for t in range(3):
    print("\n")
print("\nConfusion Matrix:")
confusion_kmeans_conc = pd.crosstab(Y_Test,Y_conc,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_kmeans_conc)


# Aftar that, PCA analysis is used to diagonalize the covariance of the new feature using the formule (S=U^-1 * sigma* U)
# sigma : covariance matrix of the PCA output 
# U is the matrix of Eigenvectors 
# S is a diagonal matrix containing the Eigenvalues

# In[48]:


# Concatenated feature matrix can be passed to (features_diagonalization) function and the diagonalized covariance of the new
# features is returned
import pandas as pd
from pandas import DataFrame

diagonalized_covariance = features_diagonalization(X_Train_conc)
df= pd.DataFrame(diagonalized_covariance)
display(df)


# * The highest values are across the digonal of the matrix.

# # <font color=navy>Comparison of Models</font>
# * <font size=3>Different Models can be compared with each other according to accuracy,time of training and time of prediction as shown in the following table:</font>
#     
# |        Model type      |    Features generation |   Time of training   |  Time of prediction  |  Accuracy  |
# |------------------------|------------------------|----------------------|----------------------| -----------------
# |  K-means 1 cluster     |Autoencoder 10 features |       0.2149 sec     |       0.2204 sec     |    68.1 %
# |                        |Autoencoder 20 features |       0.243 sec      |       0.2417 sec     |    73.5 %
# |                        |   Centroid features    |       0.1048 sec     |       0.1505 sec     |    39.1 %
# |                        |           dct          |      0.1867 sec      |       0.203 sec      |    59.0 %
# |                        |           pca          |      0.2049 sec      |       0.4456 sec     |    74.6 %
# |  K-means 2 cluster     |Autoencoder 10 features |       0.6302 sec     |       0.4122 sec     |    72.4 %
# |                        |Autoencoder 20 features |       0.6786 sec     |       0.4124 sec     |    78.1 %
# |                        |   Centroid features    |       0.211 sec      |       0.2994 sec     |    47.4 %
# |                        |           dct          |       0.7688 sec     |       0.2694 sec     |    63.8 %
# |                        |           pca          |      0.9964 sec      |       0.2717 sec     |    80.0 %
# |  K-means 4 cluster     |Autoencoder 10 features |       1.1509 sec     |       0.7499 sec     |    77.7 %
# |                        |Autoencoder 20 features |       1.669 sec      |       0.8415 sec     |    80.8 %
# |                        |   Centroid features    |       0.4457 sec     |       0.5447 sec     |    56.4 %
# |                        |           dct          |       1.7464 sec     |       0.5254 sec     |    70.3 %
# |                        |           pca          |       2.0397 sec     |       0.4357 sec     |    83.6 %
# |  K-means 8 cluster     |Autoencoder 10 features |       1.3082 sec     |       1.4513 sec     |    79.3 %
# |                        |Autoencoder 20 features |       1.7365 sec     |       1.522 sec      |    85.4 %
# |                        |   Centroid features    |       0.7509 sec     |       1.0431 sec     |    64.0 %
# |                        |           dct          |        1.7679 sec    |       0.8271 sec     |    75.0 %
# |                        |           pca          |       2.3743 sec     |       0.8552 sec     |    89.6 %
# |  K-means 16 cluster    |Autoencoder 10 features |       1.9183 sec     |       2.7627 sec     |    80.4 %
# |                        |Autoencoder 20 features |       1.8665 sec     |       2.804 sec      |    88.1 %
# |                        |   Centroid features    |       1.5426 sec     |       1.976 sec      |    70.5 %
# |   | Concatenating centroid and encoder features |       1.9795 sec     |       1.7749 sec     |    82.0 %
# |                        |          dct           |       2.2373 sec     |       1.6846 sec     |    78.2 %
# |                        |          pca           |        2.6695 sec    |       1.8107 sec     |    91.7 %
# |   | Concatenated dct and pca  features          |        3.395 sec     |       1.6362 sec     |    91.2 %
# |        1 GMM           |Autoencoder 10 features |       0.7902 sec     |       0.2434 sec     |    70.1 %
# |                        |Autoencoder 20 features |       0.5094 sec     |       0.2465 sec     |    73.5 %
# |                        |   Centroid features    |       0.3852 sec     |       0.1796 sec     |    39.1 %
# |                        |          dct           |       0.4935 sec     |       0.1967 sec     |    59.0 %
# |                        |          pca           |       0.8466 sec     |       0.1948 sec     |    74.6 %
# |        2 GMM           |Autoencoder 10 features |       3.0359 sec     |       0.3434 sec     |    69.6 %
# |                        |Autoencoder 20 features |       3.5069 sec     |       0.373 sec      |    76.2 %
# |                        |   Centroid features    |       0.7093 sec     |       0.3169 sec     |    45.8 %
# |                        |          dct           |       1.8323 sec     |       0.2372 sec     |    65.1 %
# |                        |          pca           |        6.0392 sec    |       0.279 sec      |    79.9 %
# |        4 GMM           |Autoencoder 10 features |       8.2123 sec     |       0.7396 sec     |    76.2 %
# |                        |Autoencoder 20 features |       9.5174 sec     |       0.7264 sec     |    77.9 %
# |                        |   Centroid features    |       1.8107 sec     |       0.4924 sec     |    53.4 %
# |                        |          dct           |        5.58 sec      |       0.6094 sec     |    70.9 %
# |                        |          pca           |        7.2452 sec    |       0.4895 sec     |    83.3 %
# |  SVM - linear kernals  |Autoencoder 10 features |       4.4774 sec     |       0.0679 sec     |    82.7 %
# |                        |Autoencoder 20 features |       5.7885 sec     |       0.0626 sec     |    89.1 %
# |                        |   Centroid features    |       4.4105 sec     |       0.193 sec      |    62.0 %
# |                        |          dct           |        2.2755 sec    |       0.1475 sec     |    82.0 %
# |                        |          pca           |        4.4507 sec    |       0.2876 sec     |    90.3 %
# |SVM - non linear kernals|Autoencoder 10 features |       10.2875 sec    |       0.3847 sec     |    77.0 %
# |                        |Autoencoder 20 features |       11.4055 sec    |       0.5152 sec     |    68.7 %
# |                        |   Centroid features    |       2.4326 sec     |       0.265 sec      |    81.1 %
# |                        |          dct           |        2.5146 sec    |       0.0797 sec     |    92.1 %
# |                        |          pca           |      5.3565 sec      |       0.5247 sec     |    97.3 %

# ## <font color=navy>Conclusion:</font>
# <p>Classification of Handwriiten numbers is an important part of day to day services and industries, using machines can increase efficiency of this process, descision of which algorithm to use depends on accuracy of this algorithm and processing time of it in our problem next points were noticed:</p>
# * As the number of clusters increases the accuracy increases.
# * As the number of clusters increases computational time increases.
# * GMM gives better accuracy for the same number of clusters over the KMeans, with more computational time in dct and pca features.
# * Concatenation of the PCA and DCT didn't increase the accuracy dramatically, it almost remained the same as using PCA only.
# * Centroid features was the worst among the feature reduction algorithms in most clustering algorithms however it did a better job with non linear svm than autoencoder.
# * Using Autoencoder has the upper hand regarding its accuracy since it tailors the output features on the type of the input it was trained on.
# * As the number of output nodes in the encoder increases accuracy increases.
# * Autoencoder has relatively slower timing due to time it takes to be trained on dataset rather than being a ready mathematical transformation.
# * Non linear classification can hurt the algorithm rather than benefit from it.
# * SVM with non-linear kernel was the most powerful algorithm of classification among the algorithms for this problem regarding accuracy using pca features.
# * Autoencoder can benefit from deeping the layers but that will cause the model to be computationally to be more expensive in terms of time and memory usage.
# * As the number of epochs loops increases loss function decreases and gets slower as learning algorithm reaches the minimum point.
