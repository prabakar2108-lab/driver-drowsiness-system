from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from Datapreprocessing import prepare_data
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def build_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax'),
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    train_gen, val_gen, test_gen, num_classes = prepare_data()

    model = build_model(num_classes)
    try:
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=10,
        )
    except KeyboardInterrupt:
        print("⚠️ Training interrupted by user. Proceeding to save current model state.")
    finally:
        try:
            model.save("C:/Users/manoj/Downloads/cnn_model.h5")
            print("model_saved_successfully")
        except Exception as e:
            print(f"⚠️ Failed to save model: {e}")

    test_loss, test_acc = model.evaluate(test_gen)
    print(f"✅ Test Accuracy: {test_acc:.2f}")
    print(f"Test Loss: {test_loss:.4f}")
   
