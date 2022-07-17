import numpy as np
import os
import cv2
from imutils import paths
import shutil

from imutils import paths
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import coolwarm

from keras.preprocessing import image
from keras.applications import vgg16
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from keras import models
import itertools

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# function to extract haralick textures from an image
def extract_features_vgg16(img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    out = model.predict(x)
    return out

# Initialize keras model for VGG16 architecture
model = vgg16.VGG16(include_top=False,weights='imagenet',pooling = 'avg')

OUTPUT_DATASET =  "Cropped_Images_SEN"
FOL = "SEN"
RESULTS = "Results_heatmap"
# NImages = ["NX_NY_Images_1", "length_125","length_100","NX_NY_Images_2","NX_NY_Images_3","NX_NY_Images_4",
#             "NX_NY_Images_5","NX_NY_Images_6","NX_NY_Images_7",
#             "NX_NY_Images_8", "NX_NY_Images_9","NX_NY_Images_10"]
NImages = ["NX_NY_Images_15"]

for split in NImages:
    print("[INFO] processing '{} folder'...".format(split))
    p= os.path.sep.join([OUTPUT_DATASET, split])
    results_DIR = os.path.sep.join([FOL, RESULTS, split])
    if not os.path.exists(results_DIR):
        os.makedirs(results_DIR)
    imagePaths = list(paths.list_images(p))
    totalImages = len(list(paths.list_images(p)))
    if totalImages == 0:
        pass
    else:
        features = np.empty([totalImages,512])    
        labels_num = []
        i=0
        for file in imagePaths:
            filename = file.split(os.path.sep)[-1]
            filename = file.split(os.path.sep)[-1]
            # curr_label = file.split(os.path.sep)[1]
            curr_label1 = filename.split('_')[0]
            curr_label2 = filename.split('_')[1]
            if curr_label1 == "DF140T":
                labels_num.append(0)
            elif curr_label1 == "DP980":
                labels_num.append(1)
                
            img = image.load_img(file, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features[i] = model.predict(x)
            i+=1    

        # j=1
        totalImages = len(features)
        if totalImages < 50:
            n_comp = totalImages
        else:
            n_comp = 50
        # PCA dimensionality reduction to 50 components
        pca = PCA(n_components=n_comp) #default in tsne code. Acts as sort of regularization. Makes computation of distance matrix cheaper
        x_pca = pca.fit_transform(features) #this model can handle millions of points!

        # Compute the eigenvalues and eigenvectors of the Principal Components
        PC_comp = 1 
        
        e_values = pca.explained_variance_ # These are the eigenvalues of the principal components.
                                           # Each eigenvalue expresses the importance of  each principal component.
                
        e_vectors = np.absolute(pca.components_) # This is a 50x512 matrix, where each row corresponds to the eigenvector of
                                                 # of the most important principal components and expresses how much 
                                                 # each of the features influences each principal component.First row for PC1, 2nd for PC2 ...
                
        n_major_features = e_vectors[PC_comp,:].shape[0]
                                        # Get the layer outputs from the last layers of the VGG15 architecture and create a new model that is called activation_model
        layer_outputs = [layer.output for layer in model.layers[15:]]
        activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
        input_dir = p
        dirs = os.listdir(input_dir)
  #%%                      
        # Load the images and compute an Activation Heatmap for each input image
        for item in dirs:
            # load image
            img_path = os.path.join(input_dir,item)
            # img = image.load_img(img_path)
            img = image.load_img(img_path,target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
                    
            # Get the activation maps for the last layers of the VGG16 network on the specific input image
            activations = activation_model.predict(x)
            # Then keep only the activations of the last Conv layer
            last_conv_layer_activation = activations[-2]
            heatmap = np.zeros(last_conv_layer_activation[0, :, :, 0].shape,dtype = np.float32)
            for i in range(n_major_features):
                heatmap += last_conv_layer_activation[0, :, :, i]*e_vectors[PC_comp,i]
                
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap)
                                
            # Use cv2 to load the original image
            img = cv2.imread(img_path)
            # Resize heatmap and make it RGB
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))                
            heatmap = np.uint8(255 * heatmap)
  
            # Apply the heatmap to the original image
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            out_img = heatmap * 0.4 + img
            # Save heatmap
            out_file = os.path.join(results_DIR,item)
            cv2.imwrite(out_file, out_img)
                        
                        

 
        
                        