from datetime import datetime
import os

from keras.callbacks import ModelCheckpoint

import src.config as cf
from src.models import baseline_model
from src.gen_data import data_generator, no_label_dirs


model = baseline_model()

train_set = data_generator(
    filename=os.path.join(cf.BASE_DATA_DIR, 'train_set.csv'),
    batch_size=cf.BATCH_SIZE
)
test_set = data_generator(
    filename=os.path.join(cf.BASE_DATA_DIR, 'test_set.csv'),
    batch_size=cf.BATCH_SIZE
)

model.compile(
    optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy']
)
checkpointer = ModelCheckpoint(
    filepath=cf.FMT_MODEL_CHECKPOINT.format(
        no_label_dirs(),
        2,
        datetime.now().strftime("%d-%m-%y-%H-%M-%S")
    ),
    verbose=1, save_best_only=True
)
# tensorboarder = TensorBoard(
#     log_dir='./logs', histogram_freq=0, batch_size=16,
#     write_graph=True, write_images=True
# )
NO_TRAIN_SET = sum(1 for line in open(os.path.join(cf.BASE_DATA_DIR, 'train_set.csv')))  # noqa
NO_TEST_SET = sum(1 for line in open(os.path.join(cf.BASE_DATA_DIR, 'test_set.csv')))  # noqa
model.fit_generator(
    train_set, epochs=cf.NO_EPOCHS,
    steps_per_epoch=NO_TRAIN_SET // cf.BATCH_SIZE,  # noqa
    validation_data=test_set,
    validation_steps=NO_TEST_SET // cf.BATCH_SIZE,  # noqa
    callbacks=[checkpointer]
)
model.save(
    cf.FMT_MODEL_SAVE.format(
        no_label_dirs(),
        10,
        datetime.now().strftime("%d-%m-%y-%H-%M-%S")
    )
)
