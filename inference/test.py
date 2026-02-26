from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO("../runs/detect/train6/weights/best.pt")

test_image = '../dataset/test/images/1-3-46-670589-33-1-63703718086120120200001-5487554579919763006_png_jpg.rf.9fd67251e99a47dbe83a5db6efe6c016.jpg'

url="https://www.shutterstock.com/image-illustration/kidney-stones-on-abdominal-ct-600nw-2579584785.jpg"

results = model.predict(source=url, conf=0.25, save=False)

for r in results:
    stone_count = len(r.boxes)

    print("-" * 30)
    print(f" ရလဒ်အဖြေလွှာ")
    print("-" * 30)
    print(f"တွေ့ရှိရသော ဆီးကျောက် အရေအတွက်: {stone_count} ခု")

    # ရောဂါအခြေအနေ သုံးသပ်ချက်
    if stone_count == 0:
        status = "ပုံမှန်ဖြစ်ပါသည်။ ဆီးကျောက် မတွေ့ရှိပါ။"
    elif 1 <= stone_count <= 2:
        status = "သတိပြုရန် - ဆီးကျောက် အနည်းငယ် တွေ့ရှိရပါသည်။"
    else:
        status = "သတိကြီးစွာ ထားရန် - ဆီးကျောက် အများအပြား တွေ့ရှိရပါသည်။"

    print(f" ရောဂါအခြေအနေ သုံးသပ်ချက်: {status}")
    print("-" * 30)

    im_array = r.plot(labels=False, conf=False)

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Stones Detected: {stone_count}")
    plt.show()