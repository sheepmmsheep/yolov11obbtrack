import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import tempfile
import numpy as np

# 加載 YOLO 模型
model_path = "best.pt"
model = YOLO(model_path)

# Streamlit 頁面標題
st.title("YOLO 實時追蹤")
st.sidebar.title("操作選項")
st.sidebar.write("選擇以下選項開始追蹤：")

# 設定攝像頭參數
device_index = st.sidebar.number_input("選擇攝像頭索引 (默認為 0)", min_value=0, step=1, value=0)

# 啟動攝像頭按鈕
start_tracking = st.sidebar.button("啟動追蹤")
stop_tracking = st.sidebar.button("停止追蹤")

# 創建一個函數進行攝影機推理
def webcam_tracking():
    # 打開攝像頭
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        st.error("無法打開攝影鏡頭，請檢查設備！")
        return

    # 實時處理攝像頭影像
    stframe = st.empty()  # 建立一個空框架用於顯示
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("無法捕獲影像，請重試！")
            break

        # YOLO 模型進行推理
        results = model.predict(source=frame, conf=0.35, iou=0.3, agnostic_nms=True)

        # 獲取檢測結果影像
        annotated_frame = results[0].plot()  # 繪製檢測框

        # 使用 Streamlit 實時顯示
        stframe.image(annotated_frame, channels="BGR", use_column_width=True)

    # 釋放攝像頭資源
    cap.release()

# 啟動或停止實時追蹤
if start_tracking:
    st.info("正在啟動追蹤...")
    webcam_tracking()

if stop_tracking:
    st.warning("追蹤已停止！")
