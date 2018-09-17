from datetime import datetime
import os

from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD

from models import M16
from src.gen_data import data_generator, no_label_dirs

BASE_DATA_DIR = '../datasets'  # noqa

model = M16(input_shape=(64, 64, 1))

train_set = data_generator(
    filename=os.path.join(BASE_DATA_DIR, 'train_set.csv')
)
test_set = data_generator(
    filename=os.path.join(BASE_DATA_DIR, 'test_set.csv')
)

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(
    optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']
)
checkpointer = ModelCheckpoint(
    filepath='models/checkpoints/gei_generator_modelcheckpoints_{}.{}_epochs.{}.hdf5'.format(  # noqa
        no_label_dirs,
        10,
        datetime.now().strftime("%d-%m-%y-%H-%M-%S")
    ),
    verbose=1, save_best_only=True
)
# tensorboarder = TensorBoard(
#     log_dir='./logs', histogram_freq=0, batch_size=16,
#     write_graph=True, write_images=True
# )
model.fit_generator(
    train_set, epochs=2,
    steps_per_epoch=sum(1 for line in open(os.path.join(BASE_DATA_DIR, 'train_set.csv'))) // 16,  # noqa
    validation_data=test_set,
    validation_steps=sum(1 for line in open(os.path.join(BASE_DATA_DIR, 'test_set.csv'))) // 16,  # noqa
    callbacks=[checkpointer]
)
model.save(
    "models/checkpoints/gei_generator_model_{}.{}_epochs.{}.h5".format(
        no_label_dirs,
        10,
        datetime.now().strftime("%d-%m-%y-%H-%M-%S")
    )
)
