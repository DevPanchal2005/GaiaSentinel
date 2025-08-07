import os
import cv2
import matplotlib.pyplot as plt

# Set paths
image_dir = 'data/images'   # or 'images/val'
label_dir = 'data/new_labels'   # match to image set
class_names = [
    "Plastic bag & wrapper",
    "Cigarette",
    "Unlabeled litter",
    "Bottle",
    "Bottle cap",
    "Other plastic",
    "Can",
    "Carton",
    "Cup",
    "Straw",
    "Paper",
    "Broken glass",
    "Styrofoam piece",
    "Pop tab"
]


# Visualize N random samples
N = 13
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
sample_images = image_files[N:]

for img_file in sample_images:
    img_path = os.path.join(image_dir, img_file)
    label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')

    image = cv2.imread(img_path)
    if image is None:
        print(f"❌ Could not read image: {img_path}")
        continue

    h, w = image.shape[:2]

    # Draw boxes
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, x, y, bw, bh = map(float, parts)
                x1 = int((x - bw / 2) * w)
                y1 = int((y - bh / 2) * h)
                x2 = int((x + bw / 2) * w)
                y2 = int((y + bh / 2) * h)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)
                class_label = class_names[int(cls_id)] if int(cls_id) < len(class_names) else str(cls_id)
                cv2.putText(image, class_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 5)

    else:
        print(f"⚠️ No label file found for: {img_file}")

    # Show image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.title(f"Annotations: {img_file}")
    plt.axis('off')
    plt.show()