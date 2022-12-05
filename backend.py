import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, f1_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

class Model_AI:
    def __init__(self,dataset,setting):
        self.data= dataset
        self.setting = setting
        self.AbsModel = RandomForestClassifier
        self.history = {}
        if self.setting["LogLoss"]:
            self.history["LogLoss"]={}
        if self.setting["F1"]:
            self.history["F1"]={}

    def fit(self):
        X,y = self.data.get_feature_target(self.setting["feature_list"],self.setting["target"])
        if self.setting["kfold"]:
            kf = KFold(n_splits=self.setting["K"], shuffle=True)
            fold_id=0
            for train_index, test_index in kf.split(X):
                scaler = MinMaxScaler()
                xtrain, xtest = X[train_index], X[test_index]
                xtrain = scaler.fit_transform(xtrain)
                xtest = scaler.transform(xtest)

                ytrain, ytest = y[train_index], y[test_index]

                if self.setting["pca"]:
                    PCA_Model = PCA(self.setting["pca_n"])
                    xtrain = PCA_Model.fit_transform(xtrain)

                Model = self.AbsModel()
                Model.fit(xtrain,ytrain)
                if self.setting["pca"]:
                    xtest = PCA_Model.transform(xtest)
                yhat = Model.predict(xtest)
                yhat_p = Model.predict_proba(xtest)
                if self.setting["LogLoss"]:
                    self.history["LogLoss"].update({fold_id:log_loss(ytest,yhat_p)})
                if self.setting["F1"]:
                    self.history["F1"].update({fold_id:f1_score(ytest,yhat, average='weighted')})
                fold_id+=1
        else:
            xtrain,xtest,ytrain,ytest = train_test_split(X,y, test_size = 1-self.setting["rate"])
            Model = self.AbsModel()

            if self.setting["pca"]:
                PCA_Model = PCA(self.setting["pca_n"])
                xtrain = PCA_Model.fit_transform(xtrain)

            Model.fit(xtrain,ytrain)

            if self.setting["pca"]:
                xtest = PCA_Model.transform(xtest)

            yhat = Model.predict(xtest)
            yhat_p = Model.predict_proba(xtest)
            if self.setting["LogLoss"]:
                self.history["LogLoss"].update({0:log_loss(ytest,yhat_p)})
            if self.setting["F1"]:
                self.history["F1"].update({0:f1_score(ytest,yhat, average='weighted')})
        self.model = Model
        #print(self.history)
    @property
    def get_value_metrics(self):
        return {"f1":np.mean(list(self.history["F1"].values())), "LogLoss":np.mean(list(self.history["LogLoss"].values()))}

    def plot_history(self):
        data = pd.DataFrame(self.history)
        labels = list(data.index)
        fig, ax = plt.subplots()
        title = []
        if len(labels)>1:
            labels +=["Mean"]

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars
        try:
            F1 = [i for i in data["F1"].values]
            if "Mean" in labels:
                F1 +=[np.mean(data["F1"].values)]
            rects1= ax.bar(x - width/2, F1, width, label='F1', color="green")
            title.append("F1")
        except:
            pass

        try:
            LogLoss = [i for i in data["LogLoss"].values]
            if "Mean" in labels:
                LogLoss += [np.mean(data["LogLoss"].values)]
            rects2=ax.bar(x + width/2, LogLoss, width, label='LogLoss',color="blue")
            title.append("LogLoss")
        except:
            pass
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Error')
        #ax.set_yscale("log")
        ax.set_title(f'Error of {title[0] if len(title)<1 else " and ".join(title)}')
        ax.set_xticks(x, labels)
        ax.legend()

        #ax.bar_label(rects1, padding=3)
        #ax.bar_label(rects2, padding=3)

        fig.tight_layout()
        return fig

    def extract_vector(self,features):
        feature_vector = []
        for i in features.keys():
            if i not in self.str_col:
                feature_vector.append(features[i])
        for i in self.str_col:
            feature_vector.extend(self.encoder[i].transform([[features[i]]])[0])
        return feature_vector

    def predict(self, features):
        '''
            features = { Position:..., Level:...}
        '''
        features = self.extract_vector(features)
        return self.best_model.predict([features]).reshape(1)[0]

