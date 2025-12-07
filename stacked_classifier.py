import os
import pickle
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
import pandas as pd
import seaborn as sns
import random
from tqdm import tqdm
from glob import glob

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Define all defect labels
DEFECT_LABELS = [
    'Bubble-Scatter', 'Edge-Bottle-Capped', 'Edge-Out-of-Focus', 'Lens-Inverted', 
    'Lens-Out-of-Round', 'Primary-Package-Mark', 'Lens Folded', 'Missing lens', 
    'Multiple Lenses'
]

# Set image limit
MAX_IMAGES = 25000

class DualChannelLensDataset(Dataset):
    """Dataset for lens defect images with normal and UV channels stacked together"""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels  # Multi-hot encoded tensor
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            # Load image
            img_path = self.image_paths[idx]
            full_image = Image.open(img_path).convert('L')
            width, height = full_image.size
            
            # Split into normal and UV parts
            normal_img = full_image.crop((0, 0, width, height // 2))
            uv_img = full_image.crop((0, height // 2, width, height))
            
            # Apply transformations
            if self.transform:
                normal_img = self.transform(normal_img)
                uv_img = self.transform(uv_img)
            
            # Stack channels
            stacked_img = torch.cat([normal_img, uv_img], dim=0)
            label = self.labels[idx]

            return stacked_img, label
            
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            placeholder = torch.zeros((2, 2048, 2048))
            return placeholder, self.labels[idx]

def parse_multilabel_annotations(xml_files):
    """Parse multiple XML annotation files to extract image labels"""
    image_data = {}

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for image_tag in root.findall('.//image'):
            image_name = image_tag.get('name')
            defects = set()

            for shape in image_tag:
                if shape.tag in ['polygon', 'ellipse', 'polyline', 'tag', 'box', 'points']:
                    label = shape.get('label', '')
                    if label in DEFECT_LABELS:
                        defects.add(label)

            image_data[image_name] = list(defects) if defects else []

    return image_data

def find_images(image_dirs, limit=MAX_IMAGES):
    """Find images in multiple directories"""
    image_paths = []
    for image_dir in image_dirs:
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(root, file))
                    if len(image_paths) >= limit:
                        print(f"Reached image limit ({limit})")
                        return image_paths
    return image_paths

def prepare_multilabel_data(image_dirs, xml_files, limit=MAX_IMAGES):
    """Prepare data from multiple sources for multi-label classification"""
    image_defects = parse_multilabel_annotations(xml_files)
    all_image_paths = find_images(image_dirs, limit)
    print(f"Found {len(all_image_paths)} images across directories")

    image_paths = []
    multilabel_lists = []

    for img_path in all_image_paths:
        img_name = os.path.basename(img_path)
        found = False
        
        for ann_img_name, defects in image_defects.items():
            if img_name in ann_img_name or ann_img_name in img_name:
                image_paths.append(img_path)
                multilabel_lists.append(defects)
                found = True
                break

        if not found:
            image_paths.append(img_path)
            multilabel_lists.append([])

    # Convert to multi-hot encoded vectors
    multilabel_vectors = []
    for defect_list in multilabel_lists:
        multi_hot = np.zeros(len(DEFECT_LABELS), dtype=np.float32)
        for defect in defect_list:
            if defect in DEFECT_LABELS:
                idx = DEFECT_LABELS.index(defect)
                multi_hot[idx] = 1.0
        multilabel_vectors.append(multi_hot)

    multilabel_vectors = np.array(multilabel_vectors)

    # Print statistics
    print(f"Total images: {len(image_paths)}")
    print(f"Defective images: {np.sum(np.any(multilabel_vectors, axis=1))}")
    print("Defect distribution:")
    for i, label in enumerate(DEFECT_LABELS):
        print(f"  {label}: {np.sum(multilabel_vectors[:, i])}")

    return image_paths, multilabel_vectors

class FullResolutionResNet(nn.Module):
    """Custom ResNet model for full-resolution images"""
    def __init__(self, num_classes=len(DEFECT_LABELS), pretrained=True):
        super(FullResolutionResNet, self).__init__()
        
        resnet = models.resnet50(pretrained=pretrained)
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Copy weights from pretrained model
        with torch.no_grad():
            self.conv1.weight[:, :1, :, :] = resnet.conv1.weight[:, :1, :, :]
            self.conv1.weight[:, 1:2, :, :] = resnet.conv1.weight[:, :1, :, :]
        
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.bce_logits(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

def train_multilabel_model(model, dataloaders, criterion, optimizer, scheduler=None, num_epochs=20):
    """Train the multi-label classification model"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}
    best_val_f1 = 0.0
    best_model_weights = None
    patience = 5
    no_improve = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 30)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_preds = []
            all_labels = []

            data_loader = tqdm(dataloaders[phase], desc=f'{phase.capitalize()} Batch')

            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    preds = torch.sigmoid(outputs) >= 0.5
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())

                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                data_loader.set_postfix(loss=loss.item())

            all_preds = np.concatenate(all_preds, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)

            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='weighted', zero_division=0
            )

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f}, F1: {f1:.4f}')

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_f1'].append(f1)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_f1'].append(f1)

                if scheduler is not None:
                    scheduler.step(f1)

                if f1 > best_val_f1:
                    best_val_f1 = f1
                    best_model_weights = model.state_dict().copy()
                    no_improve = 0
                    print(f"New best model saved with F1: {f1:.4f}")
                else:
                    no_improve += 1
                    print(f"No improvement for {no_improve} epochs")

        if no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    print(f"Best validation F1: {best_val_f1:.4f}")
    if best_model_weights:
        model.load_state_dict(best_model_weights)

    return model, history

def analyze_multilabel_predictions(true_labels, predicted_labels, label_names):
    """Analyze how each true label is classified"""
    n_classes = true_labels.shape[1]
    results = {}

    for i, label in enumerate(label_names):
        label_samples = true_labels[:, i] == 1
        total_samples = np.sum(label_samples)

        if total_samples == 0:
            results[label] = {
                "total_samples": 0,
                "correct_predictions": 0,
                "accuracy": 0.0,
                "misclassified_as": {}
            }
            continue

        correct_predictions = np.sum((true_labels[:, i] == 1) & (predicted_labels[:, i] == 1))
        misclassified_samples = np.where((true_labels[:, i] == 1) & (predicted_labels[:, i] == 0))[0]

        misclassifications = {}
        for sample_idx in misclassified_samples:
            wrong_predictions = np.where(predicted_labels[sample_idx] == 1)[0]
            for wrong_label_idx in wrong_predictions:
                wrong_label = label_names[wrong_label_idx]
                misclassifications[wrong_label] = misclassifications.get(wrong_label, 0) + 1

        no_prediction_count = len(misclassified_samples) - sum(misclassifications.values())
        if no_prediction_count > 0:
            misclassifications["No defect predicted"] = no_prediction_count

        accuracy = correct_predictions / total_samples
        results[label] = {
            "total_samples": int(total_samples),
            "correct_predictions": int(correct_predictions),
            "accuracy": float(accuracy),
            "misclassified_as": misclassifications
        }

    return results

def generate_detailed_classification_report(true_labels, predicted_labels, label_names):
    """Generate detailed text report for classification results"""
    results = analyze_multilabel_predictions(true_labels, predicted_labels, label_names)
    report = "DETAILED CLASSIFICATION REPORT\n============================\n\n"

    for label, data in results.items():
        total = data["total_samples"]
        correct = data["correct_predictions"]
        accuracy = data["accuracy"]

        report += f"Label: {label}\n"
        report += f"  Total samples: {total}\n"
        report += f"  Correctly classified: {correct} ({accuracy:.2%})\n"

        if total > 0 and total != correct:
            report += "  Misclassified as:\n"
            for wrong_label, count in data["misclassified_as"].items():
                report += f"    - {wrong_label}: {count} ({count/total:.2%})\n"

        report += "\n"

    overall_accuracy = accuracy_score(true_labels.flatten(), predicted_labels.flatten())
    report += f"Overall Accuracy: {overall_accuracy:.4f}\n"

    return report

def plot_multilabel_confusion_matrix(true_labels, predicted_labels, label_names):
    """Plot heatmap showing classification distribution"""
    n_classes = len(label_names)
    results = analyze_multilabel_predictions(true_labels, predicted_labels, label_names)

    matrix = np.zeros((n_classes, n_classes + 1))  # +1 for "No defect predicted"

    for i, label in enumerate(label_names):
        data = results[label]
        total_samples = data["total_samples"]

        if total_samples > 0:
            matrix[i, i] = data["correct_predictions"] / total_samples
            for wrong_label, count in data["misclassified_as"].items():
                if wrong_label == "No defect predicted":
                    matrix[i, -1] = count / total_samples
                else:
                    j = label_names.index(wrong_label)
                    matrix[i, j] = count / total_samples

    all_labels = label_names + ["No Prediction"]
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(matrix, annot=True, fmt=".2%", cmap="YlGnBu",
                    xticklabels=all_labels, yticklabels=label_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Detailed Classification Distribution')
    plt.tight_layout()
    plt.savefig('detailed_classification_matrix.png')
    print("Saved classification matrix to 'detailed_classification_matrix.png'")

    return matrix

def main():
    # Set paths - now accepts lists of directories and XML files
    image_dirs = [
        '/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/ali-tam17/H',
        '/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/ali-tam17/Q3',
        '/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/ali-tam17/Q4',
        '/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/ali-tam17/R3',
        '/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/ali-tam17/R4',
        '/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/ali-tam17/S3',
        '/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/ali-tam17/T3',
        '/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/ali-tam17/T4',
        '/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/ali-tam17/U3',
        '/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/ali-tam17/V3',
        '/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/ali-tam17/V4',
        '/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/ali-tam17/X3',

        '/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/ali-tam7/Q3',
        '/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/ali-tam7/Q4',
        '/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/ali-tam7/R3',
        '/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/ali-tam7/R4',
        '/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/ali-tam7/S3',
        '/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/ali-tam7/S4',
        '/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/ali-tam7/T3',
        '/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/ali-tam7/T4',
        '/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/ali-tam7/U3',
        '/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/ali-tam7/U4',
        '/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/ali-tam7/V3',
        '/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/ali-tam7/V4',
        '/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/ali-tam7/W3',
        '/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/ali-tam7/W4',
        '/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/ali-tam7/X3',
        '/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/ali-tam7/X4',
        '/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/ali-tam7/A',
        '/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/ali-tam7/B',
        '/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/ali-tam7/C'  # Add more directories as needed
    ]
    
    annotation_files = [
        "/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/New Annotations/Tags-Only-TAM07-CVAT-20250326.xml",
        "/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/New Annotations/Tags-Only-TAM17-CVAT-20250326.xml",
        "/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/New Annotations/TAM07-A-CVAT-20250501.xml",
        "/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/New Annotations/TAM07-B-CVAT-20250501.xml",
        "/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/New Annotations/TAM07-C-CVAT-20250508.xml",
        "/home/dell/ssd_mount/jandj-sb-uscentral/Bronze/New Annotations/TAM17-H-CVAT-20250430.xml"
    ]
    
    output_model = '/home/dell/Bhanu/20250612_StackedAllDefects_EXP.pickle'

    # Prepare data from multiple sources
    image_paths, multilabel_vectors = prepare_multilabel_data(image_dirs, annotation_files, limit=MAX_IMAGES)

    # Split into train/validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, multilabel_vectors, test_size=0.2, random_state=42
    )

    print(f"Training set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")

    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1)
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Create datasets
    train_dataset = DualChannelLensDataset(X_train, y_train, transform)
    val_dataset = DualChannelLensDataset(X_val, y_val, val_transform)

    # Data loaders
    batch_size = 8  # Smaller batch size for full resolution
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    dataloaders = {'train': train_loader, 'val': val_loader}

    # Create model
    model = FullResolutionResNet(num_classes=len(DEFECT_LABELS), pretrained=True)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Loss function and optimizer
    criterion = FocalLoss(alpha=2.0, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Train the model
    model, history = train_multilabel_model(
        model, dataloaders, criterion, optimizer, scheduler, num_epochs=10
    )

    # Save the trained model
    with open(output_model, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {output_model}")

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_f1'], label='Training F1 Score')
    plt.plot(history['val_f1'], label='Validation F1 Score')
    plt.title('F1 Score Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history saved to 'training_history.png'")

    # Evaluate on validation set
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = torch.sigmoid(outputs) >= 0.5
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Generate reports
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=DEFECT_LABELS, zero_division=0))

    detailed_report = generate_detailed_classification_report(all_labels, all_preds, DEFECT_LABELS)
    print("\nDetailed Report:")
    print(detailed_report)

    with open('detailed_classification_report.txt', 'w') as f:
        f.write(detailed_report)
    print("Detailed report saved to 'detailed_classification_report.txt'")

    # Generate visualizations
    plot_multilabel_confusion_matrix(all_labels, all_preds, DEFECT_LABELS)

    # Save classification matrix
    all_labels_with_none = DEFECT_LABELS + ["No Prediction"]
    df_matrix = pd.DataFrame(
        analyze_multilabel_predictions(all_labels, all_preds, DEFECT_LABELS),
        index=DEFECT_LABELS, columns=all_labels_with_none
    )
    df_matrix.to_csv('classification_matrix.csv')
    print("Classification matrix saved to 'classification_matrix.csv'")

if __name__ == "__main__":
    main()
