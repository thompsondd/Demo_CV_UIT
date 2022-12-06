import streamlit as st
from PIL import Image
import backend as democv

tab1, tab2 = st.tabs(["Test","Demo"])
with tab1:
    image_ip = st.file_uploader("Tải ảnh cần dự đoán loại nấm",key="data")
    if image_ip!=None:
        with open(f"./data/data_upload/image.jpg","wb+") as f:
            f.write(image_ip.getvalue())
        col1, col2 = st.columns(2)
        with col1: 
            image = Image.open("./data/data_upload/image.jpg")
            st.image(image)
        with col2: 
            st.dataframe(democv.predict("./data/data_upload/image.jpg",""))
with tab2:
    img1 = "./data/2237851965-148423.JPG"
    label1 = "Lecanoromycetes"

    img2 = "./data/2237853216-222932.jpg"
    label2 = "Pezizomycetes"

    img3 = "./data/2238154046-230379.jpg"
    label3 = "Exobasidiomycetes"

    img4 = "./data/Taphrinomycetes.jpg"
    label4 = "Taphrinomycetes"

    img5 = "./data/hoa.jpg"
    label5 = "Unknown"

    tab1_1, tab1_2, tab1_3, tab1_4, tab1_5 = st.tabs([label1,label2,label3,label4,label5])
    with tab1_1:
        col1, col2 = st.columns(2)
        with col1: st.image(Image.open(img1),caption=label1)
        with col2: st.dataframe(democv.predict(img1,label1))
    with tab1_2:
        col1, col2 = st.columns(2)
        with col1: st.image(Image.open(img2),caption=label2)
        with col2: st.dataframe(democv.predict(img2,label2))
    with tab1_3:
        col1, col2 = st.columns(2)
        with col1: st.image(Image.open(img3),caption=label3)
        with col2: st.dataframe(democv.predict(img3,label3))
    with tab1_4:
        col1, col2 = st.columns(2)
        with col1: st.image(Image.open(img4),caption=label4)
        with col2: st.dataframe(democv.predict(img4,label4))
    with tab1_5:
        col1, col2 = st.columns(2)
        with col1: st.image(Image.open(img5),caption=label5)
        with col2: st.dataframe(democv.predict(img5,label5))
