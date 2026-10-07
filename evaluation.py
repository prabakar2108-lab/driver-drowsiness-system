import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from Datapreprocessing import prepare_data

# 1. Load test data
_, _, test_gen, num_classes = prepare_data()

# 2. Load both models (make sure you saved them earlier)
cnn_model = load_model("C:/Users/manoj/Downloads/cnn_model.h5")
mobilenet_model = load_model("C:/Users/manoj/Downloads/mobilenetv2_model.h5")

# Helper to extract true labels from test data
def get_true_labels(dataset):
    if hasattr(dataset, "classes"):
        return dataset.classes

    labels = []
    for _, y in dataset:
        if hasattr(y, "numpy"):
            y = y.numpy()
        labels.append(y)

    labels = np.concatenate(labels, axis=0)
    if labels.ndim > 1 and labels.shape[1] > 1:
        labels = np.argmax(labels, axis=1)
    return labels

# Function to evaluate and plot metrics
def evaluate_model(model, test_gen, model_name):
    print(f"\n🔹 Evaluating {model_name}...")
    
    # Accuracy & Loss
    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    print(f"{model_name} Accuracy: {test_acc:.4f}")
    print(f"{model_name} Loss: {test_loss:.4f}")

    # Predictions
    y_pred_probs = model.predict(test_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = get_true_labels(test_gen)

    # Classification Report
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_true, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

# 3. Evaluate both models
evaluate_model(cnn_model, test_gen, "CNN")
evaluate_model(mobilenet_model, test_gen, "MobileNetV2")
