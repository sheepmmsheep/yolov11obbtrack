import streamlit as st
import cv2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np

# 加載 YOLO 模型
model_path = "best.pt"
model = YOLO(model_path)

# Streamlit 頁面標題
st.title("YOLO 實時追蹤 with Streamlit-WebRTC")
st.sidebar.title("操作選項")

# 信心閾值 (Confidence Threshold)
conf_threshold = st.sidebar.slider("設定信心閾值", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
# IOU 閾值
iou_threshold = st.sidebar.slider("設定IOU閾值", min_value=0.0, max_value=1.0, value=0.3, step=0.05)

# 定義影像處理回調函數
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """處理攝像頭的每一幀影像"""
    # 轉換影像為 OpenCV 格式
    img = frame.to_ndarray(format="bgr24")

    # YOLO 模型推理
    results = model.predict(source=img, conf=conf_threshold, iou=iou_threshold, agnostic_nms=True)

    # 繪製檢測框
    annotated_img = results[0].plot()

    # 返回帶有檢測框的影像
    return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# 啟動 WebRTC
webrtc_ctx = webrtc_streamer(
    key="yolo-tracking",  # 唯一標識符
    mode=WebRtcMode.SENDRECV,  # 雙向模式（接收視頻並處理後返回）
    video_frame_callback=video_frame_callback,  # 處理回調
    media_stream_constraints={"video": True, "audio": False},  # 啟用視頻輸入，禁用音頻
    async_processing=True,  # 啟用非同步處理
)

# 如果 WebRTC 正在播放，顯示提示
if webrtc_ctx.state.playing:
    st.markdown("### 正在實時追蹤中... 🔍")
    st.markdown("**請確保攝影鏡頭已啟用並正對場景**。")
else:
    st.markdown("### 點擊上方按鈕啟動實時追蹤！")
