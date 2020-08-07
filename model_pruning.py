# Model Pruning

! pip install -q tensorflow-model-optimization
! pip install codetiming
import tempfile
from codetiming import Timer
import os
import tensorflow as tf
from tensorflow import keras

## Train a MNIST model

## Using Basics steps to train a CNN model


mnist = keras.datasets.mnist #Loading MNIST model
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
  tf.keras.layers.Dense(128,activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'],
)

t = Timer()
t.start()
model.fit(
  train_images,
  train_labels,
  epochs=4,
  validation_split=0.1,
)
t.stop()

_, model_accuracy = model.evaluate(test_images, test_labels, verbose=0)

## Now, Applying Model Pruning

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
batch_size = 128
epochs = 4
validation_split = 0.2 

num_images = train_images.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.40,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=end_step)
}

pruning_model = prune_low_magnitude(model, **pruning_params)
pruning_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'],
)

logdir = tempfile.mkdtemp()

callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]

t = Timer()
t.start()
pruning_model.fit(train_images, train_labels,
                  batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                  callbacks=callbacks)
t.stop()

_, accuracy_after_pruning = pruning_model.evaluate(test_images, test_labels, verbose=0)

print('Normal model accuracy:', model_accuracy) 
print('Accuracy after Pruning:', accuracy_after_pruning)

