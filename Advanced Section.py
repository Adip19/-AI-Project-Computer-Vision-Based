import zipfile
import os
import yaml
from ultralytics import YOLO

zip_path = "/content/Coins Counter.v1i.yolov8.zip"

extract_path = "/content/Coins Counter.v1i.yolov8"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

for root, dirs, files in os.walk(extract_path):
    print("📁", root)
    for d in dirs:
        print("  📁", d)
    for f in files:
        print("  📄", f)

possible_paths = [
    os.path.join(extract_path, "train/images"),
    os.path.join(extract_path, "images"),
]

train_images_dir = None
for path in possible_paths:
    if os.path.exists(path):
        train_images_dir = path
        break

if train_images_dir is None:
    raise FileNotFoundError("Can't find the images")

print(f"Path: {train_images_dir}")

# List od the images
image_files = [f for f in os.listdir(train_images_dir) if f.endswith(".jpg")]

if len(image_files) < 50:
    raise ValueError("Didn't findmore than 50 images ")

train_250_list_path = os.path.join(extract_path, "train_250.txt")
with open(train_250_list_path, "w") as f:
    for filename in image_files[:250]:
        f.write(os.path.abspath(os.path.join(train_images_dir, filename)) + "\n")
print(f"Created: {train_250_list_path}.")

data_yaml_path = os.path.join(extract_path, "data.yaml")

with open(data_yaml_path, 'r') as file:
    data_dict = yaml.safe_load(file)

data_dict['train'] = train_50_list_path

with open(data_yaml_path, 'w') as file:
    yaml.dump(data_dict, file)


# Train model
model = YOLO("yolov8n.pt")
results = model.train(data=data_yaml_path, epochs=30, imgsz=640)

# Load the model
from ultralytics import YOLO
import os
from PIL import Image

model = YOLO("/content/runs/detect/train3/weights/best.pt")

# Value of each coin
coin_values = {
    "One": 1,
    "Two": 2,
    "Five": 5,
    "Ten": 10
}

valid_images_dir = "/content/Coins Counter.v1i.yolov8/valid/images"

# Save the result
predicted_amounts = []
ground_truth_amounts = []

# Go over the images (Valid)
for filename in os.listdir(valid_images_dir):
    if not filename.endswith(".jpg"):
        continue

    image_path = os.path.join(valid_images_dir, filename)

    #  Predict
    results = model(image_path, conf=0.25)[0]

    # Total prediction
    pred_total = 0
    for cls_idx in results.boxes.cls.tolist():
        class_name = model.names[int(cls_idx)]
        pred_total += coin_values.get(class_name, 0)
    predicted_amounts.append(pred_total)


    label_file = image_path.replace("/images/", "/labels/").replace(".jpg", ".txt")
    gt_total = 0
    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            for line in f:
                class_idx = int(line.strip().split()[0])
                class_name = model.names[class_idx]
                gt_total += coin_values.get(class_name, 0)
    else:
        print(f"Missing: {label_file}")
    ground_truth_amounts.append(gt_total)

    # Print the result of each image
    print(f"{filename}: Predicted = {pred_total}, Ground Truth = {gt_total}")

# Accuracy % (Exact Match)
if predicted_amounts:
    correct = sum([p == g for p, g in zip(predicted_amounts, ground_truth_amounts)])
    accuracy = correct / len(predicted_amounts)
    print(f"\n Accuracy (Exact Match): {accuracy * 100:.2f}%")
else:
    print("Couldn't find images to process.")