
# coding: utf-8

# # <font color='navy' size=6><b>Introdution</b></font>
# <p><font size=3>Classification of data is a main problem in machine learning, it is used in day to day problems, it can be very helpful and save time and money, in our problem here it is required to classify the the handwritten numbers into its class of number between 0-9.</font></p>
# 
# ## <font color='navy'>Clustering:</font>
# 
# <p>Clustering is an important tool in machine learning and powerful for getting useful information from data, in this assignment different clustering algorithms were used as: K-Means, GMM or SVM.</p>
# 
# ## <font color='navy'>Features reduction:</font>
# <p>Large size of features can be useful for learning algorithm however it is heavily affected by computational power of the machine so a very helpful tools can be used to reduce amount of input features while keeping most of variance and information in data using DCT or PCA.</p>
# 
# ## <font color='navy'>Outline</font>
# <p>Combining this two powerful concept can led to satisifying results for classification problem and this will be shown in the next handwritten numbers classification problem:</p>

# In[1]:


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
get_ipython().magic(u'matplotlib inline')


# ## <font color='navy'><b>Core Functions:</b></font>
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

# In[2]:


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


# ## <font color='navy'><b>Step 1:</b></font>
# * Read Data
# * Extract the features and labels
# * Standarize the features
# * Unroll the Features
# * Show an example of handwritten image

# In[3]:


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
X_Test_unroll=unroll(X_Test_std)
#show a random picture example
img_num=randint(0,X_Train.shape[0])
plt.pyplot.imshow(img.rotate(X_Train[img_num],90),origin='lower')
plt.pyplot.show()
print("Label = "+ str(Y_Train[img_num])+" image = "+ str(img_num))


# ## <font color='navy'><b>Step 2:</b></font>
# * Transform input features using DCT

# In[4]:


#dct for features
X_Train_DCT=dct_2D(X_Train_std)
X_Test_DCT=dct_2D(X_Test_std)


# ## <font color='navy'><b>Step 3:</b></font>
# * Transform input features using PCA

# In[5]:


n_comp=110
X_Train_pca,X_Train_var,pca=pca_fit(X_Train_unroll,n_comp)
X_Test_pca,X_Test_var=pca_trans(X_Test_unroll,pca)
print("Variance= "+ "%.2f" % X_Train_var+", PCA #Components = "+ str(n_comp))


# Aftar that, PCA analysis is used to diagonalize the covariance of the new feature using the formule (S=U^-1 * sigma* U)
# sigma : covariance matrix of the PCA output 
# U is the matrix of Eigenvectors 
# S is a diagonal matrix containing the Eigenvalues

# In[6]:


def covariance_diagonalization (x):
    covariance_matrix = (1/x.shape[0]) * np.dot(np.transpose(x),x)
    e_vals, e_vecs = la.eig(covariance_matrix)
    diagonal_eignvalues = inv(e_vecs) * covariance_matrix * e_vecs
    return diagonal_eignvalues
covariance_diagonalization = covariance_diagonalization (X_Train_pca)
print(covariance_diagonalization)


# * The highest values are across the digonal of the matrix.

# ## <font color='navy'><b> Kmeans Clustering on DCT Step 4.1:</b></font>
# * perform KMeans clustering with 1 cluster for each class

# In[7]:


#dct kmeans 1 cluster
tic = time.time()
kmeans_dct1=kmeans(1,10,X_Train_DCT,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# * Predict KMeans clustering with 1 cluster on test Set

# In[8]:


#dct kmeans 1 cluster Prediction 
tic = time.time()
acc_kmeans_dct1,Y_dct_KMeans1=predict_acc(X_Test_DCT,Y_Test,kmeans_dct1)
toc = time.time()
print("accuracy =",round(acc_kmeans_dct1,2),"%") 
print("elapsed time =",round(toc-tic,4),"sec\n")
print("Confusion Matrix:")
confusion_dct_KMeans1 = pd.crosstab(Y_Test, Y_dct_KMeans1,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_dct_KMeans1)


# ## <font color='navy'><b>Step 4.2:</b></font>
# * perform KMeans clustering with 2 clusters for each class

# In[9]:


#dct kmeans 2 cluster
tic = time.time()
kmeans_dct2=kmeans(2,10,X_Train_DCT,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# * Predict KMeans clustering with 2 clusters on test Set

# In[10]:


#dct kmeans 2 cluster Prediction 
tic = time.time()
acc_kmeans_dct2,Y_dct_KMeans2=predict_acc(X_Test_DCT,Y_Test,kmeans_dct2)
toc = time.time()
print("accuracy =",round(acc_kmeans_dct2,2),"%") 
print("elapsed time =",round(toc-tic,4),"sec\n")
print("Confusion Matrix:")
confusion_dct_KMeans2 = pd.crosstab(Y_Test, Y_dct_KMeans2,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_dct_KMeans2)


# ## <font color='navy'><b>Step 4.3:</b></font>
# * perform KMeans clustering with 4 clusters for each class

# In[11]:


#dct kmeans 4 cluster
tic = time.time()
kmeans_dct4=kmeans(4,10,X_Train_DCT,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# * Predict KMeans clustering with 4 clusters on test Set

# In[12]:


#dct kmeans 4 cluster Prediction 
tic = time.time()
acc_kmeans_dct4,Y_dct_KMeans4=predict_acc(X_Test_DCT,Y_Test,kmeans_dct4)
toc = time.time()
print("accuracy =",round(acc_kmeans_dct4,2),"%") 
print("elapsed time =",round(toc-tic,4),"sec\n")
print("Confusion Matrix:")
confusion_dct_KMeans4 = pd.crosstab(Y_Test, Y_dct_KMeans4,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_dct_KMeans4)


# ## <font color='navy'><b>Step 4.4:</b></font>
# * perform KMeans clustering with 8 clusters for each class

# In[13]:


#dct kmeans 8 cluster
tic = time.time()
kmeans_dct8=kmeans(8,10,X_Train_DCT,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# * Predict KMeans clustering with 8 clusters on test Set

# In[14]:


#dct kmeans 8 cluster Prediction 
tic = time.time()
acc_kmeans_dct8,Y_dct_KMeans8=predict_acc(X_Test_DCT,Y_Test,kmeans_dct8)
toc = time.time()
print("accuracy =",round(acc_kmeans_dct8,2),"%") 
print("elapsed time =",round(toc-tic,4),"sec\n")
print("Confusion Matrix:")
confusion_dct_KMeans8 = pd.crosstab(Y_Test, Y_dct_KMeans8,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_dct_KMeans8)


# ## <font color='navy'><b>Step 4.5:</b></font>
# * perform KMeans clustering with 16 clusters for each class

# In[15]:


#dct kmeans 16 cluster
tic = time.time()
kmeans_dct16=kmeans(16,10,X_Train_DCT,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# * Predict KMeans clustering with 16 clusters on test Set

# In[16]:


#dct kmeans 16 cluster Prediction 
tic = time.time()
acc_kmeans_dct16,Y_dct_KMeans16=predict_acc(X_Test_DCT,Y_Test,kmeans_dct16)
toc = time.time()
print("accuracy =",round(acc_kmeans_dct16,2),"%") 
print("elapsed time =",round(toc-tic,4),"sec\n")
print("Confusion Matrix:")
confusion_dct_KMeans16 = pd.crosstab(Y_Test, Y_dct_KMeans16,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_dct_KMeans16)


# ## <font color='navy'><b>K-Means Clustering on PCA Step 5.1:</b></font>
# * perform KMeans clustering with 1 cluster for each class

# In[17]:


#pca kmeans 1 Cluster 
tic = time.time()
kmeans_pca1=kmeans(1,10,X_Train_pca,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# * Predict KMeans clustering with 1 cluster on test Set

# In[18]:


#pca kmeans 1 Cluster Prediction
toc = time.time()
acc_kmeans_pca1,Y_pca_KMeans1=predict_acc(X_Test_pca,Y_Test,kmeans_pca1)
toc = time.time()
print("accuracy =",round(acc_kmeans_pca1,2),"%")
print("elapsed time =",round(toc-tic,4),"sec")
print("\nConfusion Matrix:")
confusion_pca_KMeans1 = pd.crosstab(Y_Test, Y_pca_KMeans1,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_pca_KMeans1)


# ## <font color='navy'><b>Step 5.2:</b></font>
# * perform KMeans clustering with 2 clusters for each class

# In[19]:


#pca kmeans 2 Cluster 
tic = time.time()
kmeans_pca2=kmeans(2,10,X_Train_pca,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# * Predict KMeans clustering with 2 clusters on test Set

# In[20]:


#pca kmeans 2 Cluster Prediction
tic = time.time()
acc_kmeans_pca2,Y_pca_KMeans2=predict_acc(X_Test_pca,Y_Test,kmeans_pca2)
toc = time.time()
print("accuracy =",round(acc_kmeans_pca2,2),"%")
print("elapsed time =",round(toc-tic,4),"sec")
print("Confusion Matrix:")
confusion_pca_KMeans2 = pd.crosstab(Y_Test, Y_pca_KMeans2,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_pca_KMeans2)


# ## <font color='navy'><b>Step 5.3:</b></font>
# * perform KMeans clustering with 4 clusters for each class

# In[21]:


#pca kmeans 4 Cluster 
tic = time.time()
kmeans_pca4=kmeans(4,10,X_Train_pca,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# * Predict KMeans clustering with 4 clusters on test Set

# In[22]:


#pca kmeans 4 Cluster Prediction
tic = time.time()
acc_kmeans_pca4,Y_pca_KMeans4=predict_acc(X_Test_pca,Y_Test,kmeans_pca4)
toc = time.time()
print("accuracy =",round(acc_kmeans_pca4,2),"%")
print("elapsed time =",round(toc-tic,4),"sec")
print("\nConfusion Matrix:")
confusion_pca_KMeans4 = pd.crosstab(Y_Test, Y_pca_KMeans4,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_pca_KMeans4)


# ## <font color='navy'><b>Step 5.4:</b></font>
# * perform KMeans clustering with 8 clusters for each class

# In[23]:


#pca kmeans 8 Cluster 
tic = time.time()
kmeans_pca8=kmeans(8,10,X_Train_pca,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# * Predict KMeans clustering with 8 clusters on test Set

# In[24]:


#pca kmeans 8 Cluster Prediction
tic = time.time()
acc_kmeans_pca8,Y_pca_KMeans8=predict_acc(X_Test_pca,Y_Test,kmeans_pca8)
toc = time.time()
print("accuracy =",round(acc_kmeans_pca8,2),"%")
print("elapsed time =",round(toc-tic,4),"sec")
print("\nConfusion Matrix:")
confusion_pca_KMeans8 = pd.crosstab(Y_Test, Y_pca_KMeans8,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_pca_KMeans8)


# ## <font color='navy'><b>Step 5.5:</b></font>
# * perform KMeans clustering with 16 clusters for each class

# In[25]:


#pca kmeans 16 Cluster 
tic = time.time()
kmeans_pca16=kmeans(16,10,X_Train_pca,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# * Predict KMeans clustering with 16 clusters on test Set

# In[26]:


#pca kmeans 16 Cluster Prediction
tic = time.time()
acc_kmeans_pca16,Y_pca_KMeans16=predict_acc(X_Test_pca,Y_Test,kmeans_pca16)
toc = time.time()
print("accuracy =",round(acc_kmeans_pca16,2),"%")
print("elapsed time =",round(toc-tic,4),"sec")
print("\nConfusion Matrix:")
confusion_pca_KMeans16 = pd.crosstab(Y_Test, Y_pca_KMeans16,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_pca_KMeans16)


# ## <font color='navy'><b> GMM Clustering on DCT Step 6.1:</b></font>
# * perform KMeans clustering with 1 cluster for each class

# In[27]:


#dct GMM 1
tic = time.time()
G_dct1=GMM(1,10,X_Train_DCT,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# * Predict KMeans clustering with 1 clusters on test Set

# In[28]:


#dct GMM 1 Predictions
tic = time.time()
acc_GMM_dct1,Y_dct_GMM1=predict_acc(X_Test_DCT,Y_Test,G_dct1)
toc = time.time()
print("accuracy =",round(acc_GMM_dct1,2),"%") 
print("elapsed time =",round(toc-tic,4),"sec")
print("\nConfusion Matrix:")
confusion_dct_GMM1 = pd.crosstab(Y_Test, Y_dct_GMM1,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_dct_GMM1)


# ## <font color='navy'><b>Step 6.2:</b></font>
# * perform KMeans clustering with 2 cluster for each class

# In[29]:


#dct GMM 2
tic = time.time()
G_dct2=GMM(2,10,X_Train_DCT,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# * Predict KMeans clustering with 2 clusters on test Set

# In[30]:


#dct GMM 2 Predictions
tic = time.time()
acc_GMM_dct2,Y_dct_GMM2=predict_acc(X_Test_DCT,Y_Test,G_dct2)
toc = time.time()
print("accuracy =",round(acc_GMM_dct2,2),"%") 
print("elapsed time =",round(toc-tic,4),"sec") 
print("\nConfusion Matrix:")
confusion_dct_GMM2 = pd.crosstab(Y_Test, Y_dct_GMM2,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_dct_GMM2)


# ## <font color='navy'><b>Step 6.3:</b></font>
# * perform KMeans clustering with 4 cluster for each class

# In[31]:


#dct GMM 4
tic = time.time()
G_dct4=GMM(4,10,X_Train_DCT,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# * Predict KMeans clustering with 4 clusters on test Set

# In[32]:


#dct GMM 4 Predictions
tic = time.time()
acc_GMM_dct4,Y_dct_GMM4=predict_acc(X_Test_DCT,Y_Test,G_dct4)
toc = time.time()
print("accuracy =",round(acc_GMM_dct4,2),"%")  
print("elapsed time =",round(toc-tic,4),"sec")   
print("\nConfusion Matrix:")
confusion_dct_GMM4 = pd.crosstab(Y_Test, Y_dct_GMM4,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_dct_GMM4)


# ## <font color='navy'><b>GMM Clustering on PCA Step 7.1:</b></font>
# * perform KMeans clustering with 1 cluster for each class

# In[33]:


#pca GMM  1
tic = time.time()
G_pca1=GMM(1,10,X_Train_pca,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# * Predict KMeans clustering with 1 clusters on test Set

# In[34]:


#pca GMM 1 Predictions
tic = time.time()
acc_GMM_pca1,Y_pca_GMM1=predict_acc(X_Test_pca,Y_Test,G_pca1)
toc = time.time()
print("accuracy =",round(acc_GMM_pca1,2),"%") 
print("elapsed time =",round(toc-tic,4),"sec") 
print("\nConfusion Matrix:")
confusion_pca_GMM1 = pd.crosstab(Y_Test, Y_pca_GMM1,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_pca_GMM1)


# ## <font color='navy'><b>Step 7.2:</b></font>
# * perform KMeans clustering with 2 cluster for each class

# In[35]:


#pca GMM  2
tic = time.time()
G_pca2=GMM(2,10,X_Train_pca,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# * Predict KMeans clustering with 2 clusters on test Set

# In[36]:


#pca GMM 2 Predictions
tic = time.time()
acc_GMM_pca2,Y_pca_GMM2=predict_acc(X_Test_pca,Y_Test,G_pca2)
toc = time.time()
print("accuracy =",round(acc_GMM_pca2,2),"%") 
print("elapsed time =",round(toc-tic,4),"sec") 
print("\nConfusion Matrix:")
confusion_pca_GMM2 = pd.crosstab(Y_Test, Y_pca_GMM2,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_pca_GMM2)


# ## <font color='navy'><b>Step 7.3:</b></font>
# * perform KMeans clustering with 4 cluster for each class

# In[37]:


#pca GMM  4
tic = time.time()
G_pca4=GMM(4,10,X_Train_pca,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# * Predict KMeans clustering with 4 clusters on test Set

# In[38]:


#pca GMM 4 Predictions
tic = time.time()
acc_GMM_pca4,Y_pca_GMM4=predict_acc(X_Test_pca,Y_Test,G_pca4)
toc = time.time()
print("accuracy =",round(acc_GMM_pca4,2),"%") 
print("elapsed time =",round(toc-tic,4),"sec") 
print("\nConfusion Matrix:")
confusion_pca_GMM4 = pd.crosstab(Y_Test, Y_pca_GMM4,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_pca_GMM4)


# ## <font color='navy'><b>Classification using Linear SVM with DCT Step 8:</b></font>
# * perform Linear SVM Classification

# In[39]:


#dct Linear SVM
tic = time.time()
svm_dct_lin = svm.SVC(kernel='linear')
svm_dct_lin.fit(X_Train_DCT,Y_Train)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec") 


# * Predict with Linear SVM on test Set

# In[40]:


#dct Linear SVM Prediction
tic = time.time()
acc_dct_svm=accuracy_score(Y_Test,svm_dct_lin.predict(X_Test_DCT))*100
toc = time.time()
print('accuracy = ',round(acc_dct_svm,2),"%")
print("elapsed time =",round(toc-tic,4),"sec") 
print("\nConfusion Matrix:")
confusion_svm_dct_lin = pd.crosstab(Y_Test, svm_dct_lin.predict(X_Test_DCT),                                    rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_svm_dct_lin)


# ## <font color='navy'><b>Classification using Non-Linear kernel SVM with DCT Step 9:</b></font>
# * perform Non-Linear SVM Classification

# In[41]:


#dct non-Linear SVM
tic = time.time()
svm_dct_nlin = svm.SVC(kernel='poly')
svm_dct_nlin.fit(X_Train_DCT,Y_Train)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec") 


# * Predict with Non-Linear SVM on test Set

# In[42]:


#dct non-Linear SVM Prediction
tic = time.time()
acc_dct_nsvm=accuracy_score(Y_Test,svm_dct_nlin.predict(X_Test_DCT))*100
toc = time.time()
print('accuracy = ',round(acc_dct_nsvm,2),"%")
print("elapsed time =",round(toc-tic,4),"sec") 
print("\nConfusion Matrix:")
confusion_svm_dct_nlin = pd.crosstab(Y_Test,svm_dct_nlin.predict(X_Test_DCT),                                    rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_svm_dct_nlin)


# ## <font color='navy'><b>Classification using Linear SVM with PCA Step 10:</b></font>
# * perform Linear SVM Classification

# In[43]:


#pca Linear SVM
tic = time.time()
svm_pca_lin = svm.SVC(kernel='linear')
svm_pca_lin.fit(X_Train_pca,Y_Train)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec") 


# * Predict with Linear SVM on test Set

# In[44]:


#pca Linear SVM Prediction
tic = time.time()
acc_pca_svm=accuracy_score(Y_Test,svm_pca_lin.predict(X_Test_pca))*100
toc = time.time()
print('accuracy = ',round(acc_pca_svm,2),"%")
print("elapsed time =",round(toc-tic,4),"sec")
print("\nConfusion Matrix:")
confusion_svm_pca_lin = pd.crosstab(Y_Test,svm_pca_lin.predict(X_Test_pca),                                    rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_svm_pca_lin)


# ## <font color='navy'><b>Classification using Non-Linear Kernel SVM with PCA Step 11:</b></font>
# * perform Non-Linear SVM Classification

# In[45]:


#pca non-Linear SVM
tic = time.time()
svm_pca_nlin = svm.SVC(kernel='poly')
svm_pca_nlin.fit(X_Train_pca,Y_Train)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec") 


# * Predict with Non-Linear SVM on test Set

# In[46]:


#pca non-Linear SVM Prediction
tic = time.time()
acc_pca_nsvm=accuracy_score(Y_Test,svm_pca_nlin.predict(X_Test_pca))*100
toc = time.time()
print('accuracy = ',round(acc_pca_nsvm,2),"%")
print("elapsed time =",round(toc-tic,4),"sec") 
print("\nConfusion Matrix:")
confusion_svm_pca_nlin = pd.crosstab(Y_Test,svm_pca_nlin.predict(X_Test_pca),                                    rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_svm_pca_nlin)


# ## <font color='navy'><b>Concatenation of DCT and PCA features Step 12:</b></font>
# * perform Concatenation

# In[47]:


X_Train_conc=np.concatenate((X_Train_pca,X_Train_DCT),axis=1)
X_Test_conc=np.concatenate((X_Test_pca,X_Test_DCT),axis=1)


# * Perform K-Means clustering with 16 cluster

# In[48]:


#concatenated K-Means
tic = time.time()
kmeans_conc=kmeans(16,10,X_Train_conc,1000)
toc = time.time()
print("elapsed time =",round(toc-tic,4),"sec")


# * Predict on the conactenated features

# In[49]:


#concatenated kmeans Predictions
tic = time.time()
acc_kmeans_conc,Y_conc=predict_acc(X_Test_conc,Y_Test,kmeans_conc)
toc = time.time()
print("accuracy =",round(acc_kmeans_conc,2),"%")
print("elapsed time =",round(toc-tic,4),"sec") 
print("\nConfusion Matrix:")
confusion_kmeans_conc = pd.crosstab(Y_Test,Y_conc,rownames=['Actual'], colnames=['Predicted'], margins=True)
display(confusion_kmeans_conc)


# # <font color=navy>Comparison of Models</font>
# * <font size=3>Different Models Processing Time and accuracies  are shown in the following table :</font>
#     
# |        Model type      |   Time of training | Time of prediction  |  Accuracy  |
# |------------------------|--------------------|----------------------| -----------------
# |  dct kmeans 1 cluster  |      0.1575 sec    |       0.1835 sec     |    59.0 %
# |  dct kmeans 2 cluster  |      0.8128 sec    |       0.3320 sec     |    63.8 %
# |  dct kmeans 4 cluster  |      1.3830 sec    |       0.5454 sec     |    70.3 %
# |  dct kmeans 8 cluster  |      1.8654 sec    |       0.8386 sec     |    75.0 %
# |  dct kmeans 16 cluster |      2.2383 sec    |       1.6592 sec     |    78.2 %
# |  pca kmeans 1 Cluster  |      0.1909 sec    |       0.3548 sec     |    74.6 %
# |  pca kmeans 2 Cluster  |      1.1443 sec    |       0.2988 sec     |    80.1 %
# |  pca kmeans 4 Cluster  |      2.0916 sec    |       0.4627 sec     |    84.0 %
# |  pca kmeans 8 Cluster  |      2.3143 sec    |       0.8857 sec     |    89.5 %
# |  pca kmeans 16 Cluster |      2.6674 sec    |       1.6466 sec     |    91.8 %
# |  dct GMM 1             |      0.3606 sec    |       0.1775 sec     |    59.0 %
# |  dct GMM 2             |      1.7436 sec    |       0.2937 sec     |    65.1 %
# |  dct GMM 4             |      5.6600 sec    |       0.7784 sec     |    70.9 %
# |  pca GMM 1             |      0.9161 sec    |       0.1598 sec     |    74.5 %
# |  pca GMM 2             |      5.0936 sec    |       0.2963sec      |    80.0 %
# |  pca GMM 4             |      9.4406 sec    |       0.5454 sec     |    83.0 %
# |  dct Linear SVM        |      2.4367 sec    |       0.1107 sec     |    82.0 %
# |  dct non-Linear SVM    |      2.3865 sec    |       0.0749 sec     |    92.1 %
# |  pca Linear SVM        |      5.1712 sec    |       0.2524 sec     |    90.3 %
# |  pca non-Linear SVM    |      5.5565 sec    |       0.5134 sec     |    97.4 %
# |  concatenated K-Means  |      2.7132 sec    |       1.8375 sec     |    92.0 %

# ## <font color=navy>Conclusion:</font>
# <p>Classification of Handwriiten numbers is an important part of day to day services and industries, using machines can increase efficiency of this process, descision of which algorithm to use depends on accuracy of this algorithm and processing time of it in our problem next points were noticed:</p>
# * As the number of clusters increases the accuracy increases.
# * As the number of clusters increases computational time increases.
# * GMM gives better accuracy for the same number of clusters over the KMeans, with more computational time.
# * Concatenation of the PCA and DCT didn't increase the accuracy dramatically, it almost remained the same as using PCA only.
# * SVM with non-linear kernel was the most powerful algorithm of classification among the algorithms for this problem regarding accuracy.
# 
