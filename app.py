import streamlit as st
import cv2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np

st.title("YOLO Real-Time Tracking with YOLO.track")
st.sidebar.title("Options")

# Load YOLO Model
model_path = "best.pt"  # Update with the actual model path
try:
    model = YOLO(model_path)
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")

# Sidebar Controls
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.35, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.3, 0.05)

# Video Frame Callback
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")  # Convert frame to OpenCV format
    
    # Optionally resize for faster processing
    img_resized = cv2.resize(img, (640, 480))
    
    # Perform tracking
    try:
        results = model.track(source=img_resized, conf=conf_threshold, iou=iou_threshold, persist=True)
        if len(results) > 0:
            annotated_img = results[0].plot()
        else:
            annotated_img = img_resized
    except Exception as e:
        st.error(f"Error during tracking: {e}")
        annotated_img = img_resized
    
    return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# WebRTC Streamer
webrtc_ctx = webrtc_streamer(
    key="yolo-tracking",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# User Instructions
if webrtc_ctx.state.playing:
    st.markdown("### Real-Time Tracking in Progress... 🔍")
    st.markdown("**Ensure the webcam is enabled and positioned correctly.**")
else:
    st.markdown("### Click the button above to start real-time tracking!")
