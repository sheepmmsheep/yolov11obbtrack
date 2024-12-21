import streamlit as st
import cv2
from PIL import Image
import torch

from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import numpy as np
device = 'cpu'
if not hasattr(st, 'obb'):
    st.model = torch.hub.load('ultralytics/yolov11', 'custom', path="best.pt", _verbose=False)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
st.title("YOLO 实时追踪 with YOLO.track")
st.sidebar.title("操作选项")

conf_threshold = st.sidebar.slider("設置信心閥值", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
iou_threshold = st.sidebar.slider("設置IOU閥值", min_value=0.0, max_value=1.0, value=0.3, step=0.05)

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")  

    results = st.model.track(source=img, conf=conf_threshold, iou=iou_threshold, persist=True)

    annotated_img = results[0].plot()

    return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# 启动 WebRTC
webrtc_ctx = webrtc_streamer(
    key="yolo-tracking",  # 唯一标识符
    mode=WebRtcMode.SENDRECV,  # 双向模式（接收视频并处理后返回）
    video_frame_callback=video_frame_callback,  # 视频处理回调
    media_stream_constraints={"video": True, "audio": False},  # 启用视频输入，禁用音频
    async_processing=True,  # 启用异步处理
)

# 显示状态信息
if webrtc_ctx.state.playing:
    st.markdown("### 正在实时追踪中... 🔍")
    st.markdown("**请确保摄像头已启用并正对场景。**")
else:
    st.markdown("### 点击上方按钮启动实时追踪！")
