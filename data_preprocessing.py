import os
import splitfolders
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def prepare_data(
    raw_dataset_path="C:/Users/manoj/Downloads/dataset/train",
    split_dataset_path="dataset_split",
    img_size=(224, 224),
    batch_size=32,
    seed=123,
):
    """Prepare train, validation, and test datasets for model training."""
    if not os.path.exists(split_dataset_path):
        if not os.path.exists(raw_dataset_path):
            raise FileNotFoundError(
                f"Raw dataset path not found: {raw_dataset_path}. Please update raw_dataset_path."
            )

        print("Splitting dataset into train/val/test...")
        splitfolders.ratio(
            raw_dataset_path,
            output=split_dataset_path,
            ratio=(0.7, 0.15, 0.15),
            seed=seed,
            group_prefix=None,
        )
        print("✅ Dataset split into train/val/test (70/15/15)")
    else:
        print("✅ Dataset already split and ready")

    train_dir = os.path.join(split_dataset_path, "train")
    val_dir = os.path.join(split_dataset_path, "val")
    test_dir = os.path.join(split_dataset_path, "test")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError(
            f"Expected split dataset folders not found under {split_dataset_path}."
        )

    num_classes = len(
        [name for name in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, name))]
    )

    try:
        train_gen = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            seed=seed,
            image_size=img_size,
            batch_size=batch_size,
            label_mode="categorical",
            shuffle=True,
        )

        val_gen = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            seed=seed,
            image_size=img_size,
            batch_size=batch_size,
            label_mode="categorical",
            shuffle=False,
        )

        test_gen = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            seed=seed,
            image_size=img_size,
            batch_size=batch_size,
            label_mode="categorical",
            shuffle=False,
        )

        normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
        train_gen = train_gen.map(lambda x, y: (normalization_layer(x), y))
        val_gen = val_gen.map(lambda x, y: (normalization_layer(x), y))
        test_gen = test_gen.map(lambda x, y: (normalization_layer(x), y))

        print("✅ Preprocessing complete — using modern TensorFlow API")

    except Exception as e:
        print(f"⚠️ Modern API failed: {e}")
        print("Falling back to ImageDataGenerator...")

        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=15,
            zoom_range=0.15,
            brightness_range=[0.8, 1.2],
            horizontal_flip=True,
        )

        val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

        train_gen = train_datagen.flow_from_directory(
            train_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode="categorical",
        )
        val_gen = val_test_datagen.flow_from_directory(
            val_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode="categorical",
        )
        test_gen = val_test_datagen.flow_from_directory(
            test_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False,
        )

        num_classes = train_gen.num_classes
        print("✅ Preprocessing complete — using ImageDataGenerator")

    return train_gen, val_gen, test_gen, num_classes


if __name__ == "__main__":
    prepare_data()
