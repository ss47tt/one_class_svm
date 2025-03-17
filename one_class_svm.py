import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import os
from tqdm import tqdm  # Progress bar
import matplotlib.pyplot as plt
import seaborn as sns

# Load pretrained ResNet50 model (without classifier)
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove last layer
resnet.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to extract deep features with progress bar
def extract_features(folder):
    features = []
    filenames = []
    
    for filename in tqdm(os.listdir(folder), desc=f"Extracting features from {folder}"):
        img_path = os.path.join(folder, filename)
        
        # Skip non-image files
        if not (filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif"))):
            continue  

        try:
            img = Image.open(img_path).convert("RGB")
            img = transform(img).unsqueeze(0)

            with torch.no_grad():
                feat = resnet(img).flatten().numpy()  # Extract deep features
            
            features.append(feat)
            filenames.append(filename)  # Only store valid image filenames

        except (UnidentifiedImageError, OSError) as e:
            print(f"Skipping file {filename}: {e}")

    return np.array(features), filenames  # Return valid features and filenames

# Load and extract deep features with progress bars
X_train, _ = extract_features("data/train")
X_test_cat, test_filenames_cat = extract_features("data/test/cat")  # Normal class
X_test_other, test_filenames_other = extract_features("data/test/other")  # Anomaly class

# Normalize and train One-Class SVM
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test_cat = scaler.transform(X_test_cat)
X_test_other = scaler.transform(X_test_other)

print("\nTraining One-Class SVM...")
oc_svm = OneClassSVM(kernel="rbf", gamma=0.001, nu=0.08)
oc_svm.fit(X_train)

# Predict with progress bar
print("\nPredicting anomalies...")
predictions_cat = [oc_svm.predict([x])[0] for x in tqdm(X_test_cat, desc="Processing cat images")]
predictions_other = [oc_svm.predict([x])[0] for x in tqdm(X_test_other, desc="Processing other images")]

# Prepare labels
true_labels = [1] * len(predictions_cat) + [-1] * len(predictions_other)  # 1 = normal (cat), -1 = anomaly (other)
pred_labels = predictions_cat + predictions_other

# Compute confusion matrix
cm = confusion_matrix(true_labels, pred_labels, labels=[1, -1])

# Save confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

# Save classification report
report = classification_report(true_labels, pred_labels, target_names=['Normal (Cat)', 'Anomaly (Other)'])
with open("classification_report.txt", "w") as f:
    f.write(report)

# Save individual results
results = []
for filename, pred in zip(test_filenames_cat + test_filenames_other, pred_labels):
    status = "Normal (Cat)" if pred == 1 else "Anomaly (Other)"
    results.append(f"{filename}: {status}")
    print(f"{filename}: {status}")

with open("results.txt", "w") as f:
    f.write("\n".join(results))

# Display results
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)