import streamlit as st
import cv2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import numpy as np

model_path = "best.pt"  
model = YOLO(model_path)

st.title("YOLO 實時追蹤 with YOLO.track")
st.sidebar.title("操作選項")

conf_threshold = st.sidebar.slider("設置信心閥值", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
iou_threshold = st.sidebar.slider("設置IOU閥值", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")  # 转换为 OpenCV 格式
    results = model.track(source=img, conf=conf_threshold, iou=iou_threshold, persist=True)
    annotated_img = results[0].plot()
    return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")
class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = process(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")
webrtc_ctx = webrtc_streamer(
    key="yolo-tracking",  
    mode=WebRtcMode.SENDRECV,  
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,  
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)

if webrtc_ctx.state.playing:
    st.markdown("### 正在实时追踪中... 🔍")
    st.markdown("**请确保摄像头已启用并正对场景。**")
else:
    st.markdown("### 点击上方按钮启动实时追踪！")
