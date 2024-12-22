import streamlit as st
import cv2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np

# 加载 YOLO 模型
model_path = "best.pt"  # 替换为你的模型路径
model = YOLO(model_path)

# Streamlit 页面标题
st.title("YOLO 实时追踪 with YOLO.track")
st.sidebar.title("操作选项")

# 信心阈值和 IOU 阈值
conf_threshold = st.sidebar.slider("设置信心阈值", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
iou_threshold = st.sidebar.slider("设置IOU阈值", min_value=0.0, max_value=1.0, value=0.3, step=0.05)

# 定义视频处理函数
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """处理摄像头的每一帧影像"""
    img = frame.to_ndarray(format="bgr24")  # 转换为 OpenCV 格式

    # 使用 YOLO.track 进行跟踪
    results = model.track(source=img, conf=conf_threshold, iou=iou_threshold, persist=True)

    # 绘制检测框
    annotated_img = results[0].plot()

    # 返回带有检测框的影像
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
