"""Run from project root: streamlit run app.py"""
import json
from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st
import torch
from torchvision.transforms.functional import pil_to_tensor
from src.model import load_checkpoint
from src.config import CLASSES
from src.visualize import draw_detections

st.set_page_config(page_title='VOC Object Detector', layout='wide')
st.title('Object Detection · PASCAL VOC 2012')
st.write('Upload an image to locate objects from 20 everyday categories.')
policy_path = Path('artifacts/deployment.json')
if not policy_path.exists():
    st.info('Run the notebook through model export first. It creates artifacts/best.pt and deployment.json.')
    st.stop()
policy = json.loads(policy_path.read_text())
checkpoint = Path('artifacts/best.pt')
if not checkpoint.exists():
    st.error('artifacts/best.pt is missing. Finish the notebook export section.')
    st.stop()

@st.cache_resource
def detector(path, modification_time, min_size, max_size):
    model, cfg, _ = load_checkpoint(path, 'cpu')
    model.transform.min_size = (min_size,)
    model.transform.max_size = max_size
    return model

model = detector(str(checkpoint), checkpoint.stat().st_mtime_ns, policy['min_size'], policy['max_size'])
threshold = st.slider('Confidence threshold', .05, .95, .5, .05)
upload = st.file_uploader('JPG, PNG or WebP image', type=['jpg', 'jpeg', 'png', 'webp'])
if upload:
    try:
        image = ImageOps.exif_transpose(Image.open(upload)).convert('RGB')
        if image.width * image.height > 16_000_000:
            st.error('Please use an image smaller than 16 megapixels.')
            st.stop()
        tensor = pil_to_tensor(image).float() / 255.
        with st.spinner('Detecting objects...'), torch.inference_mode():
            prediction = model([tensor])[0]
        left, right = st.columns(2)
        left.image(image, caption='Input', use_container_width=True)
        right.image(draw_detections(image, prediction, threshold), caption='Detected objects', use_container_width=True)
        rows = [{'class': CLASSES[int(label)], 'confidence': float(score),
                 'xmin': float(box[0]), 'ymin': float(box[1]),
                 'xmax': float(box[2]), 'ymax': float(box[3])}
                for box, score, label in zip(prediction['boxes'], prediction['scores'], prediction['labels'])
                if float(score) >= threshold]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.download_button('Download detections as JSON', json.dumps(rows, indent=2),
                           file_name='detections.json', mime='application/json')
    except (OSError, ValueError) as exc:
        st.error(f'Unable to process this image: {exc}')
st.caption('Research demo. Predictions can miss or misclassify objects.')
