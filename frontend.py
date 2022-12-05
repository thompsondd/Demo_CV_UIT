import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import backend as ml
import sklearn.datasets as datasets
import math
from PIL import Image

#1. Markdown
st.markdown("""
# Hệ thống AI dự báo
""")
#data = st.file_uploader("Tải file dữ liệu về lương để AI dự đoán",key="data")
setting={}
class count_section:
    def __init__(self):
        self.number = 0
    def __call__(self):
        self.number+=1
        return self.number
image_ip = st.file_uploader("Tải ảnh cần dự đoán loại nấm",key="data")
if image_ip!=None:
    get_num_section = count_section()

    # Select input feature
    
    
    image = Image.open(image_ip)
    st.image(image)
    if st.button("Run",key="run"):
        pass
        '''
        if n_selected_features==0:
            st.error("Please select at lease a feature")
        elif (setting["F1"] or setting["LogLoss"])==False:
            st.error("Please select a metric for evaluation")
        else:
            feature_selected = [ i for i in get_feature_inputs.keys() if get_feature_inputs[i]]
            if len(feature_selected)<1:
                st.error("Please select at least a feature")
            else:
                setting.update({"feature_list":feature_selected,"target":target})
                model = ml.Model_AI(dataset, setting)
                model.fit()

                stats = model.get_value_metrics
                for i in stats.keys():
                    st.write(f"{i} : {stats[i]}")
                st.pyplot(model.plot_history())
    '''