from ultralytics import YOLO
import gradio as gr
import cv2
import numpy as np

# Load trained model
model = YOLO("../runs/detect/train6/weights/best.pt")

def detect_kidney_stone(image):

    # Run prediction
    results = model.predict(source=image, conf=0.4)

    for r in results:
        stone_count = len(r.boxes)

        # Calculate average confidence
        if stone_count > 0:
            conf_scores = r.boxes.conf.cpu().numpy()
            avg_conf = float(conf_scores.mean())
        else:
            avg_conf = 0.0

        # Medical interpretation
        if stone_count == 0:
            status = "ပုံမှန်ဖြစ်ပါသည်။ ဆီးကျောက် မတွေ့ရှိပါ။"
        elif 1 <= stone_count <= 2:
            status = "သတိပြုရန် - ဆီးကျောက် အနည်းငယ် တွေ့ရှိရပါသည်။"
        else:
            status = "သတိကြီးစွာ ထားရန် - ဆီးကျောက် အများအပြား တွေ့ရှိရပါသည်။"

        # Plot detection result
        im_array = r.plot(labels=False, conf=False)
        im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)

        report = f"""
        ----------------------------
        🩺 ဆီးကျောက် စစ်ဆေးမှု ရလဒ်
        ----------------------------
        တွေ့ရှိရသော ဆီးကျောက် အရေအတွက်: {stone_count} ခု
        ပျမ်းမျှ ယုံကြည်မှု (Confidence): {avg_conf:.2f}
        ရောဂါအခြေအနေ သုံးသပ်ချက်: {status}
        ----------------------------
        """

        return im_rgb, report


# Build Gradio Interface
interface = gr.Interface(
    fn=detect_kidney_stone,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Image(label="Detection Result"),
        gr.Textbox(label="Medical Report")
    ],
    title="Kidney Stone Detection System",
    description="Upload CT Image to Detect Kidney Stones"
)

interface.launch()