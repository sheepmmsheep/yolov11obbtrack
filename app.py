import streamlit as st
from ultralytics import YOLO
import tempfile
from PIL import Image
import os
import cv2
import numpy as np
import time

def process_image(image_source, model):
    """處理單張圖片"""
    try:
        results = model.predict(image_source)
        return Image.fromarray(results[0].plot())
    except Exception as e:
        st.error(f"圖片處理錯誤: {str(e)}")
        return None

def process_video(video_path, model):
    """處理影片檔案"""
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_output.close()
    
    try:
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 設定輸出影片編碼器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))
        
        # 進度條
        progress_bar = st.progress(0)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 處理影格
            results = model.predict(frame)
            annotated_frame = results[0].plot()
            
            # 寫入影格
            out.write(annotated_frame)
            
            # 更新進度條
            frame_count += 1
            progress = int(frame_count / total_frames * 100)
            progress_bar.progress(progress)
        
        # 釋放資源
        cap.release()
        out.release()
        
        return temp_output.name
        
    except Exception as e:
        st.error(f"影片處理錯誤: {str(e)}")
        if os.path.exists(temp_output.name):
            os.unlink(temp_output.name)
        return None

def main():
    st.title("車道線偵測系統")
    
    # 初始化模型
    @st.cache_resource
    def load_model():
        return YOLO('best.pt')
    
    try:
        model = load_model()
    except Exception as e:
        st.error(f"模型載入錯誤: {str(e)}")
        return
    
    # 選擇輸入類型
    input_type = st.radio("選擇輸入類型:", ["圖片", "影片", "即時影像"])
    
    if input_type == "圖片":
        uploaded_file = st.file_uploader("上傳圖片", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption='原始圖片', use_column_width=True)
                
                if st.button('開始偵測'):
                    with st.spinner('處理中...'):
                        result_image = process_image(image, model)
                        if result_image:
                            st.image(result_image, caption='偵測結果', use_column_width=True)
            except Exception as e:
                st.error(f"圖片處理錯誤: {str(e)}")
    
    elif input_type == "影片":
        uploaded_file = st.file_uploader("上傳影片", type=['mp4', 'avi', 'mov'])
        
        if uploaded_file is not None:
            try:
                temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_input.write(uploaded_file.read())
                temp_input.close()
                
                st.video(temp_input.name)
                
                if st.button('開始偵測'):
                    with st.spinner('處理中... 這可能需要一些時間'):
                        st.text('正在處理影片，請稍候...')
                        result_video = process_video(temp_input.name, model)
                        if result_video:
                            st.video(result_video)
                            
                if os.path.exists(temp_input.name):
                    os.unlink(temp_input.name)
                    
            except Exception as e:
                st.error(f"影片處理錯誤: {str(e)}")
    
    else:  # 即時影像
        st.write("即時影像辨識")
        
        # 即時影像處理
        cap = cv2.VideoCapture(0)  # 使用預設攝影機
        
        if not cap.isOpened():
            st.error("無法開啟攝影機")
            return
            
        # 建立影像顯示區域
        image_placeholder = st.empty()
        stop_button = st.button('停止')
        
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("無法讀取影像")
                break
                
            # 處理影格
            results = model.predict(frame)
            annotated_frame = results[0].plot()
            
            # 轉換顏色空間
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # 顯示影格
            image_placeholder.image(annotated_frame_rgb, caption='即時偵測', use_column_width=True)
            
            # 控制更新頻率
            time.sleep(0.1)
        
        # 釋放資源
        cap.release()

if __name__ == '__main__':
    main()
