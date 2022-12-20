from tensorflow import keras
import tensorflow as tf
import joblib
import cv2 as cv
from skimage import feature
import pandas as pd
import os
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras import Model

model_folder = r"./model_and_encoder"

def get_model_path(model):
    return os.path.join(model_folder,model)
SVM_hist_path = "hinge_l2_10_LinearSVC_hist.pkl"
Softmax_hist_path = "multinomial_lbfgs_LogisticRegression_hist.pkl"

Softmax_hog_path = "multinomial_sag_LogisticRegression_hog.pkl"
SVM_hog_path = "squared_hinge_l2_1LinearSVC_hog.pkl"

SVM_vgg16_path = "VGG16_LinearSVC_l2_hinge.sav"
Softmax_vgg16_path = "VGG16_softmax_lbfgs.sav"

SVM_vgg19_path = "VGG19_LinearSVC_l2_hinge.sav"
Softmax_vgg19_path = "VGG19_softmax_lbfgs.sav"

encoder_ = "encoder_hog.pkl"

def calHist(img):
    hist = cv.calcHist([img],[0],None, [256],[0,256])
    size = img.shape[0]*img.shape[1]
    hist = hist / size
    return hist

def calHOG(img):
    img =cv.resize(img, (256,128))
    (hog, hog_image) = feature.hog(img, orientations=9,  
                                 pixels_per_cell=(16, 16), cells_per_block=(2, 2),  
                                 block_norm='L2-Hys', visualize=True, transform_sqrt=True)
    return hog

def build_model(input_shape,model):
    base_model = model(include_top=True, input_shape=input_shape, weights='imagenet')
    
    base_model.trainable = False
    model = Model(inputs=base_model.inputs, 
                           outputs=base_model.layers[-2].output,
                           name="main_model")
    return model

W,H = 224,224

def calModel(img, model):
    img = cv.resize(img,(W,H))
    with tf.device('GPU:0'):
        input_d = tf.convert_to_tensor([img])
        #print(input_d.shape)
    return model(input_d).numpy()

def calModel16(img):
    model = build_model((W,H,3),keras.applications.VGG16)
    feature = calModel(img,model)
    return feature
def calModel19(img):
    model = build_model((W,H,3),keras.applications.VGG19)
    feature = calModel(img,model)
    return feature

def extract_feature(file_path,calF,c=1):
    img = cv.imread(file_path,c)
    return calF(img.copy())

SVM_hist = joblib.load(open(get_model_path(SVM_hist_path), 'rb'))
Softmax_hist = joblib.load(open(get_model_path(Softmax_hist_path), 'rb'))

Softmax_hog = joblib.load(open(get_model_path(Softmax_hog_path), 'rb'))
SVM_hog = joblib.load(open(get_model_path(SVM_hog_path), 'rb'))

SVM_vgg16 = joblib.load(open(get_model_path(SVM_vgg16_path), 'rb'))
Softmax_vgg16 = joblib.load(open(get_model_path(Softmax_vgg16_path), 'rb'))

SVM_vgg19 = joblib.load(open(get_model_path(SVM_vgg19_path), 'rb'))
Softmax_vgg19 = joblib.load(open(get_model_path(Softmax_vgg19_path), 'rb'))

encoder = joblib.load(open(get_model_path(encoder_), 'rb'))


def predict(image_path, true_label):
    feature = extract_feature(image_path,calHOG,0)
    
    y_hat_svm_hog = SVM_hog.predict([feature.reshape(-1)])
    y_hat_svm_hog = encoder.classes_[y_hat_svm_hog]
        
    y_hat_softmax_hog = Softmax_hog.predict([feature.reshape(-1)])
    y_hat_softmax_hog = encoder.classes_[y_hat_softmax_hog]
    #------------------------------------------------------------------
    feature = extract_feature(image_path,calHist,0)
    
    y_hat_svm_hist = SVM_hist.predict([feature.reshape(-1)])
    y_hat_svm_hist = encoder.classes_[y_hat_svm_hist]
        
    y_hat_softmax_hist = Softmax_hist.predict([feature.reshape(-1)])
    y_hat_softmax_hist = encoder.classes_[y_hat_softmax_hist]
    #----------------------------------------------------------------------
    feature = extract_feature(image_path,calModel16,1)
    
    y_hat_svm_vgg16 = SVM_vgg16.predict([feature.reshape(-1)])
    y_hat_svm_vgg16 = encoder.classes_[y_hat_svm_vgg16]
        
    y_hat_softmax_vgg16 = Softmax_vgg16.predict([feature.reshape(-1)])
    y_hat_softmax_vgg16 = encoder.classes_[y_hat_softmax_vgg16]
    #--------------------------------------------------------------------
    feature = extract_feature(image_path,calModel19,1)
    
    y_hat_svm_vgg19 = SVM_vgg19.predict([feature.reshape(-1)])
    y_hat_svm_vgg19 = encoder.classes_[y_hat_svm_vgg19]
        
    y_hat_softmax_vgg19 = Softmax_vgg19.predict([feature.reshape(-1)])
    y_hat_softmax_vgg19 = encoder.classes_[y_hat_softmax_vgg19]
    #-------------------------------------------------------------------------------------
    table = {"Hist":{"Softmax Regression":y_hat_softmax_hist[0],"SVM":y_hat_svm_hist[0]},
            "HOG":{"Softmax Regression":y_hat_softmax_hog[0],"SVM":y_hat_svm_hog[0]},
            "VGG16":{"Softmax Regression":y_hat_softmax_vgg16[0],"SVM":y_hat_svm_vgg16[0]},
            "VGG19":{"Softmax Regression":y_hat_softmax_vgg19[0],"SVM":y_hat_svm_vgg19[0]},
        }
    print(f"True label:{true_label}\n\tHOG\n\t\tSoftmax predict:\t{y_hat_softmax_hog}\n\t\tSVM predict:\t\t{y_hat_svm_hog}\
    \n\tHist\n\t\tSoftmax predict:\t{y_hat_softmax_hist}\n\t\tSVM predict:\t\t{y_hat_svm_hist}\
    \n\tVGG16\n\t\tSoftmax predict:\t{y_hat_softmax_vgg16}\n\t\tSVM predict:\t\t{y_hat_svm_vgg16}\
    \n\tVGG19\n\t\tSoftmax predict:\t{y_hat_softmax_vgg19}\n\t\tSVM predict:\t\t{y_hat_svm_vgg19}")
    df = pd.DataFrame(table).T
    dff = df.style.apply(lambda x: ["background-color:lime" if i==true_label else "background-color:" for i in x ])
    return dff
