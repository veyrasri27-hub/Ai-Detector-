# app.py - Enhanced Deepfake Detector v2: Multi-Model + Artifacts + Video Flow Analysis
# Supports AI images (Midjourney/DALL-E/Flux), deepfakes, face-swaps, lip-sync proxies, video synth, hybrids
# Run with: pip install gradio transformers torch torchvision opencv-python pillow numpy pandas plotly
#          python app.py

import gradio as gr
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
import cv2
import numpy as np
import os
import sys
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# === Configuration ===
MODEL_NAMES = [
    "prithivMLmods/Deep-Fake-Detector-v2-Model",  # General deepfake
    "dima806/deepfake_vs_real_image_detection",    # Deepfake vs real images
    "capcheck/ai-image-detection"                  # Modern AI gen (Midjourney/DALL-E/Flux)
]
FAKE_CONFIDENCE_THRESHOLD = 0.65
TEMP_DIR = tempfile.gettempdir()

print("\n" + "="*80)
print("Loading multi-models... (may take 3-5 minutes, ~2GB RAM/GPU)")
print("="*80)

device = "cuda" if torch.cuda.is_available() else "cpu"
processors = {}
models = {}
LABELS = {}

try:
    for model_name in MODEL_NAMES:
        print(f"Loading {model_name}...")
        processors[model_name] = AutoImageProcessor.from_pretrained(model_name)
        models[model_name] = AutoModelForImageClassification.from_pretrained(model_name)
        models[model_name].to(device)
        models[model_name].eval()
        LABELS[model_name] = models[model_name].config.id2label
        print(f"✓ {model_name} loaded on {device.upper()}")
    print("="*80 + "\n")
except Exception as e:
    print(f"❌ Model loading error: {e}")
    sys.exit(1)

def compute_artifacts(pil_img):
    img_np = np.array(pil_img)
    if img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Laplacian variance (sharpness/blur)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # FFT high frequency ratio (AI often has distinct spectrum)
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude = 20 * np.log(np.clip(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]), 1e-10, None))
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    center_block = magnitude[cy-32:cy+32, cx-32:cx+32]
    low_freq_energy = np.sum(center_block)
    total_energy = np.sum(magnitude)
    high_freq_ratio = 1 - (low_freq_energy / total_energy) if total_energy > 0 else 0.0
    
    # Noise estimation
    noise_level = np.std(cv2.Laplacian(gray.astype(np.float32), cv2.CV_64F))
    
    return {
        "laplacian_variance": float(lap_var),
        "high_freq_ratio": float(high_freq_ratio),
        "noise_level": float(noise_level)
    }

def ensemble_predict(pil_img):
    all_fake_probs = []
    all_real_probs = []
    model_preds = []
    
    for model_name in MODEL_NAMES:
        inputs = processors[model_name](images=pil_img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = models[model_name](**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
        
        pred_idx = np.argmax(probs)
        pred_label = LABELS[model_name][pred_idx]
        model_preds.append(pred_label)
        
        prob_real = 0.0
        prob_fake = 0.0
        for idx, label in LABELS[model_name].items():
            lower_label = label.lower()
            if any(word in lower_label for word in ["real", "authentic"]):
                prob_real = probs[idx]
            if any(word in lower_label for word in ["fake", "deepfake", "synthetic", "ai", "generated"]):
                prob_fake = probs[idx]
        
        all_fake_probs.append(prob_fake)
        all_real_probs.append(prob_real)
    
    avg_fake = np.mean(all_fake_probs)
    avg_real = np.mean(all_real_probs)
    ensemble_pred = "🟥 FAKE / DEEPFAKE" if avg_fake > FAKE_CONFIDENCE_THRESHOLD else "🟢 REAL"
    
    return ensemble_pred, avg_real, avg_fake, all_fake_probs, model_preds

def detect_image(image):
    if image is None:
        return "No image", 0.0, 0.0, {}, "Unknown", None
    
    try:
        img = image.convert("RGB")
        artifacts = compute_artifacts(img)
        ensemble_pred, avg_real, avg_fake, model_fakes, model_preds = ensemble_predict(img)
        
        result = f"**Ensemble Prediction:** {ensemble_pred}\n\n"
        result += "**Model Scores:**\n"
        for i, name in enumerate(MODEL_NAMES):
            short_name = name.split("/")[-1]
            result += f"- **{short_name}**: {model_preds[i]} ({model_fakes[i]:.1%} Fake)\n"
        result += f"\n**Averages:** Real {avg_real:.2%} | Fake **{avg_fake:.2%}**\n\n"
        result += "**CV Artifacts:**\n"
        result += f"| Metric | Value | Indicator |\n"
        result += f"|---|---|---|\n"
        result += f"| Laplacian Var (sharpness) | {artifacts['laplacian_variance']:.1f} | {'🟡 High=AI sharp' if artifacts['laplacian_variance'] > 200 else '✅ Normal'} |\n"
        result += f"| High Freq Ratio | {artifacts['high_freq_ratio']:.3f} | {'🟡 Low=AI smoothed' if artifacts['high_freq_ratio'] < 0.75 else '✅ Natural'} |\n"
        result += f"| Noise Level | {artifacts['noise_level']:.1f} | {'🟡 Unusual' if artifacts['noise_level'] > 50 else '✅ Typical'} |\n\n"
        
        # Category inference
        if avg_fake > 0.65:
            if artifacts['high_freq_ratio'] < 0.72 or artifacts['laplacian_variance'] > 250:
                category = "🖼️ **AI-Generated Image** (Midjourney/DALL-E/Flux/Stable Diffusion)"
            elif artifacts['laplacian_variance'] > 180:
                category = "😵 **Face-Swap** (Deepfacelab/Roop)"
            else:
                category = "🎥 **Video Synthesis / Hybrid** (Sora/Runway + face)"
        else:
            category = "✅ **Real / Low Risk**"
        
        result += f"**Detected Type:** {category}\n"
        result += f"\n*Threshold: >{FAKE_CONFIDENCE_THRESHOLD:.0%} Fake → Alert*"
        
        return result, avg_real, avg_fake, artifacts, category, None
        
    except Exception as e:
        return f"❌ Error: {str(e)[:100]}", 0.0, 0.0, {}, "Error", None

def compute_optical_flow_stats(frames):
    if len(frames) < 2:
        return 0.0, 0.0
    flow_mags = []
    for i in range(1, len(frames)):
        prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
        if prev_pts is not None:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
            if curr_pts is not None:
                good_pts = curr_pts[status == 1]
                if len(good_pts) > 0:
                    dx = good_pts[:, 0] - prev_pts[status == 1, 0]
                    dy = good_pts[:, 1] - prev_pts[status == 1, 1]
                    mag = np.sqrt(dx**2 + dy**2)
                    flow_mags.append(np.median(mag))
    std_mag = np.std(flow_mags) if flow_mags else 0.0
    mean_mag = np.mean(flow_mags) if flow_mags else 0.0
    return std_mag, mean_mag

def load_video_safely(video_path):
    try:
        if not os.path.exists(video_path):
            return None, "Video not found"
        safe_temp = os.path.join(TEMP_DIR, f"deepfake_vid_{os.getpid()}.mp4")
        shutil.copy2(video_path, safe_temp)
        return safe_temp, None
    except Exception as e:
        return None, f"❌ Load error: {str(e)[:100]}"

def extract_frames(video_path, num_frames=16, progress=gr.Progress()):
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "Cannot open video"
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // num_frames)
        count = 0
        frame_idx = 0
        while cap.isOpened() and frame_idx < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if count % step == 0:
                frames.append(frame)
                progress((frame_idx + 1) / num_frames, desc="Extracting frames...")
                frame_idx += 1
            count += 1
        cap.release()
        return frames, None
    except Exception as e:
        return None, f"❌ Extract error: {str(e)[:100]}"

def detect_faces_in_frame(frame):
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        if len(faces) > 0:
            face_regions = [frame[y:y+h, x:x+w] for x, y, w, h in faces]
            return face_regions
        return [frame]
    except:
        return [frame]

def detect_video(video_path, threshold, progress=gr.Progress()):
    if video_path is None:
        return "No video", 0.0, None, {}
    
    safe_video, error = load_video_safely(video_path)
    if error:
        return error, 0.0, None, {}
    
    try:
        frames, error = extract_frames(safe_video, num_frames=16, progress=progress)
        if error:
            return error, 0.0, None, {}
        
        flow_std, flow_mean = compute_optical_flow_stats(frames)
        fake_probs = []
        artifacts_list = []
        frame_faces_count = []
        analyzed_count = 0
        
        for i, frame in enumerate(frames):
            progress(i / len(frames), desc="Analyzing frames...")
            face_regions = detect_faces_in_frame(frame)
            frame_faces_count.append(len(face_regions))
            
            frame_artifacts = []
            for face_region in face_regions[:2]:  # Limit per frame
                face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_rgb)
                _, _, frame_fake, _, _, _ = detect_image(face_pil)
                fake_probs.append(frame_fake)
                art = compute_artifacts(face_pil)
                frame_artifacts.append(art)
                analyzed_count += 1
            
            if frame_artifacts:
                avg_art_frame = {k: np.mean([a[k] for a in frame_artifacts]) for k in frame_artifacts[0]}
                artifacts_list.append(avg_art_frame)
        
        if not fake_probs:
            return "No analyzable content", 0.0, None, {}
        
        avg_fake = np.mean(fake_probs)
        max_fake = np.max(fake_probs)
        verdict = "🟥 Likely DEEPFAKE VIDEO" if avg_fake > threshold else "🟢 Likely REAL VIDEO"
        
        # Overall artifacts
        if artifacts_list:
            avg_artifacts = {k: np.mean([a[k] for a in artifacts_list]) for k in artifacts_list[0]}
        else:
            avg_artifacts = {}
        
        result = f"**Verdict:** {verdict}\n\n"
        result += f"**Stats:**\n- Avg Fake: **{avg_fake:.2%}**\n- Max Fake: {max_fake:.2%}\n"
        result += f"- Analyzed: {analyzed_count} faces/frames\n- Threshold: {threshold:.2%}\n\n"
        result += f"**Motion Analysis:**\n- Flow Std: {flow_std:.3f} {'🟡 High=Unnatural motion (synth/re-enact)' if flow_std > 2 else '✅ Consistent'}\n"
        result += f"- Avg Motion: {flow_mean:.2f}\n\n"
        
        if avg_artifacts:
            result += "**Avg Artifacts:**\n"
            result += f"- Lap Var: {avg_artifacts['laplacian_variance']:.1f}\n"
            result += f"- High Freq: {avg_artifacts['high_freq_ratio']:.3f}\n"
        
        # Category
        motion_fake = flow_std > 2.5 or flow_mean < 0.5
        if avg_fake > threshold:
            if avg_artifacts.get('high_freq_ratio', 0.8) < 0.72:
                cat = "🎬 **AI Video Gen** (Sora/Runway)"
            elif motion_fake:
                cat = "💃 **Full-Body Reenactment** or **Lip-Sync Fake** (Wav2Lip)"
            else:
                cat = "😵 **Face-Swap / Hybrid**"
        else:
            cat = "✅ **Real Video**"
        result += f"**Likely Type:** {cat}"
        
        df_plot = pd.DataFrame({
            'analyzed_item': [f"Frame {i//max(1,frame_faces_count[i//len(frame_faces_count)])} Face {i%2+1}" for i in range(len(fake_probs))],
            'fake_prob': fake_probs
        })
        fig = px.line(df_plot, x='analyzed_item', y='fake_prob', 
                      title='Fake Prob per Face/Frame', markers=True)
        fig.update_layout(xaxis_tickangle=-45)
        
        return result, avg_fake, fig, avg_artifacts
        
    except Exception as e:
        return f"❌ Analysis error: {str(e)[:100]}", 0.0, None, {}
    finally:
        if safe_video and os.path.exists(safe_video):
            os.remove(safe_video)

# === Gradio Interface ===
with gr.Blocks(title="Veera's Enhanced Deepfake Detector v2") as demo:
    gr.Markdown("# 🚨 **Veera's Enhanced Deepfake Detector v2**")
    gr.Markdown("""
    **Multi-Model Ensemble + CV Artifacts + Motion Analysis**
    - **Models:** Deepfake-v2 + Image-Deepfake + AI-Image-Det (Midjourney/DALL-E/Flux)
    - Detects: AI images, face-swaps, video synth, reenactment, hybrids
    """)
    
    with gr.Tabs():
        with gr.TabItem("🖼️ Image Analysis"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(label="Upload Image", type="pil", sources=["upload", "clipboard", "webcam"])
                    img_btn = gr.Button("🔍 Analyze", variant="primary", scale=2)
                with gr.Column():
                    img_output = gr.Markdown(label="Detailed Results")
                    img_artifacts = gr.JSON(label="Raw Artifacts", visible=False)
            
            with gr.Row():
                img_prob_real = gr.Number(label="Ensemble Real Prob", interactive=False)
                img_prob_fake = gr.Number(label="Ensemble Fake Prob", interactive=False)
                img_category = gr.Markdown(label="Detected Category")
        
        with gr.TabItem("🎥 Video Analysis"):
            with gr.Row():
                with gr.Column(scale=2):
                    vid_input = gr.Video(label="Upload Video")
                with gr.Column(scale=1):
                    threshold_slider = gr.Slider(0.3, 0.9, FAKE_CONFIDENCE_THRESHOLD, 0.05, 
                                                 label="Fake Threshold", info="Video alert threshold")
                    vid_btn = gr.Button("🔍 Analyze Video", variant="primary")
            
            vid_output = gr.Markdown(label="Results")
            vid_prob_avg = gr.Number(label="Avg Fake Prob", interactive=False)
            vid_plot = gr.Plot(label="Fake Prob Timeline")
            vid_artifacts = gr.JSON(label="Avg Video Artifacts", visible=False)
    
    def process_image(image):
        result, prob_real, prob_fake, artifacts, category, _ = detect_image(image)
        return result, prob_real, prob_fake, artifacts, category
    
    def process_video(video, threshold):
        result, avg_prob, fig, artifacts = detect_video(video, threshold)
        return result, avg_prob, fig, artifacts
    
    img_btn.click(
        process_image,
        inputs=img_input,
        outputs=[img_output, img_prob_real, img_prob_fake, img_artifacts, img_category],
        show_progress="full"
    )
    
    vid_btn.click(
        process_video,
        inputs=[vid_input, threshold_slider],
        outputs=[vid_output, vid_prob_avg, vid_plot, vid_artifacts],
        show_progress="full"
    )

print("\n" + "="*80)
print("🚀 Launching Enhanced App at http://127.0.0.1:7860")
print("="*80)
demo.launch(share=False, show_error=True, server_name="0.0.0.0", server_port=7860)