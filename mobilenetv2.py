from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from Datapreprocessing import prepare_data
import os
from pathlib import Path


def build_model(num_classes):
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    preds = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=preds)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
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
            model.save("C:/Users/manoj/Downloads/mobilenetv2_model.h5")
            print("model_saved_successfully")
        except Exception as e:
            print(f"⚠️ Failed to save model: {e}")
    
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"✅ Test Accuracy: {test_acc:.2f}")
        
       
    
    
