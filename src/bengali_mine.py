import time
startNB = time.time()   # Time the execution of the notebook
import tensorflow as tf, os
import tensorflow.keras.backend as K
import efficientnet.tfkeras as efn
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, accuracy_score
import albumentations as albu, cv2, gc
print('TensorFlow version = {}'.format(tf.__version__))

# CONFIGURE GPUs
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if len(gpus) == 1:
    strategy = tf.distribute.OneDeviceStrategy(device='/gpu:0')
else:
    strategy = tf.distribute.MirroredStrategy()

# ENABLE MIXED PRECISION for speed
# tf.config.optimizer.set_jit(True)
tf.config.optimizer.set_experimental_options({'auto_mixed_precision': True})
print('Mixed precision enabled')  # When float32 needed, define dtype explicitly

# Logging directory
logging_dir = '../logging/'

# VERSION MAJOR and MINOR for logging
mm = 1
rr = 0

# BEGIN LOG FILE
f = open(os.path.join(logging_dir, f'log-{mm}-{rr}.txt'), 'a')
print('Logging to {}'.format(f.name))
f.write('#############################\n')
f.write(f'Trial mm={mm}, rr={rr}\n')
f.write('efNetB4, batch_size=512, seed=42, 64x64, fold=0, LR 1e-3 with 0.75 decay\n')
f.write('#############################\n')
f.close()

# TODO: Consolidate hyperparameters
BATCH_SIZE = 512
DIM = 64

data_dir = '../data/raw/'

train = []
# for x in [0, 1, 2, 3]:
for x in [0]:  # 25% of data
    f = 'train_image_data_%i.parquet' % x
    print(f, end='')
    img = pd.read_parquet(os.path.join(data_dir, f))  # pd df
    img = img.iloc[:, 1:].values.reshape((-1, 137, 236, 1))  # np array
    img2 = np.zeros((img.shape[0], DIM, DIM, 1), dtype='float32')
    for j in range(img.shape[0]):
        img2[j, :, :, 0] = cv2.resize(img[j, ],
                                      (DIM, DIM),
                                      interpolation=cv2.INTER_AREA)
        if j % 1000 == 0:
            print(j, ', ', end='')
    print()
    img2 = (255 - img2) / 255.  # normalize
    train.append(img2)

X_train = np.concatenate(train)
print('Train shape', X_train.shape)

del img, img2, train
_ = gc.collect()   # TODO: ?

row = 3
col = 4
plt.figure(figsize=(20, (row / col) * 12))
for x in range(row * col):
    plt.subplot(row, col, x + 1)
    plt.imshow(X_train[x, :, :, 0])
plt.show()

train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
y_train = train.iloc[:, 1:4].values[:len(X_train)]
print('Labels\nGrapheme Root, Vowel Diacritic, Consonant Diacritic')
y_train


class DataGenerator(tf.keras.utils.Sequence):  # TODO: Sequence
    'Generates data for Keras'

    def __init__(self,
                 X,
                 y,
                 list_IDs,
                 batch_size=BATCH_SIZE,
                 shuffle=False,
                 augment=False,
                 labels=True,
                 cutmix=False,
                 yellow=False):

        self.X = X
        self.y = y
        self.augment = augment
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.labels = labels
        self.cutmix = cutmix
        self.yellow = yellow
        self.on_epoch_end()  # TODO: What does this do?

    def __len__(self):
        'Denotes the number of batches per epoch'
        ct = len(self.list_IDs) // self.batch_size
        ct += int((len(self.list_IDs) % self.batch_size) != 0)  # One batch if data left
        return ct

    def __getitem__(self, index):
        'Generates one batch of data'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(indexes)
        if self.augment:
            X = self.__augment_batch(X)
        if self.labels:
            return X, [y[:, 0:168], y[:, 168:179], y[:, 179:186]]
            # TODO: Examine the data to see labels
        else:
            return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)
            # TODO: Grok indexes

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        X = self.X[self.list_IDs[indexes],]
        if self.yellow:
            X = np.ones((len(indexes), DIM, DIM, 1))
        y = np.zeros((len(indexes), 186))
        for j in range(len(indexes)):
            y[j, int(self.y[self.list_IDs[indexes[j]], 0])] = 1
            y[j, 168 + int(self.y[self.list_IDs[indexes[j]], 1])] = 1
            y[j, 179 + int(self.y[self.list_IDs[indexes[j]], 2])] = 1
            # TODO: Grok

        if self.cutmix:
            for j in range(len(indexes)):
                # CHOOSE RANDOM CENTER
                yy = np.random.randint(0, DIM)
                xx = np.random.randint(0, DIM)
                z = np.random.choice(self.list_IDs)
                # TODO: Grok

                # CHOOSE RANDOM WIDTH AND HEIGHT
                h = np.random.randint(DIM // 2 - DIM // 16, DIM // 2 + DIM // 16)  # TODO: Grok
                w = np.random.randint(DIM // 2 - DIM // 16, DIM // 2 + DIM // 16)

                # CUT AND MIX IMAGES
                ya = max(0, yy - h // 2)
                yb = min(DIM, yy + h // 2)
                xa = max(0, xx - w // 2)
                xb = min(DIM, xx + w // 2)
                X[j, ya:yb, xa:xb, 0] = self.X[z, ya:yb, xa:xb, 0]

                # CUT AND MIX LABELS
                r = (yb - ya) * (xb - xa) / DIM / DIM
                y2 = np.zeros((1, 186))
                y2[0, int(self.y[z, 0])] = 1
                y2[0, 168 + int(self.y[z, 1])] = 1
                y2[0, 179 + int(self.y[z, 2])] = 1
                y[j, ] = (1 - r) * y[j, ] + r * y2[0, ]

        return X, y

    def __random_transform(self, img):
        composition = albu.Compose([
            albu.OneOf([
                albu.ShiftScaleRotate(rotate_limit=8,
                                      scale_limit=0.16,
                                      shift_limit=0,
                                      border_mode=0,
                                      value=0,
                                      p=0.5),
                albu.CoarseDropout(max_holes=16,
                                   max_height=DIM // 10,
                                   max_width=DIM // 10,
                                   fill_value=0,
                                   p=0.5)
            ], p=0.5),
            albu.ShiftScaleRotate(rotate_limit=0,
                                  scale_limit=0.,
                                  shift_limit=0.12,
                                  border_mode=0,
                                  value=0,
                                  p=0.5)
        ])
        return composition(image=img)['image']
        # TODO: Is this a signature?

    def __augment_batch(self, img_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i, ] = self.__random_transform(img_batch[i, ])
        return img_batch

print('Cutmix augmentation with first image in all yellow')
gen = DataGenerator(X_train,
                    y_train,
                    np.arange(len(X_train)),
                    shuffle=True,
                    augment=True,
                    batch_size=BATCH_SIZE,
                    cutmix=True,
                    yellow=True)
row = 3
col = 4
plt.figure(figsize=(20, (row / col) * 12))
for batch in gen:
    for j in range(row * col):
        plt.subplot(row, col, j + 1)
        plt.imshow(batch[0][j, :, :, 0])
    plt.show()
    break

model_dir = '../models/'
weights_file = 'efnB4.h5'


def build_model():
    inp = tf.keras.Input(shape=(DIM, DIM, 1))
    inp2 = tf.keras.layers.Concatenate()([inp, inp, inp])
    # 3 channels
    base_model = efn.EfficientNetB4(weights=None,
                                    include_top=False,
                                    input_shape=(DIM, DIM, 3))
    base_model.load_weights(os.path.join(model_dir, weights_file))

    x = base_model(inp2)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x1 = tf.keras.layers.Dense(168,
                               activation='softmax',
                               name='x1',
                               dtype='float32')(x)
    # Explicit due to mixed precision setup at top
    x2 = tf.keras.layers.Dense(11,
                               activation='softmax',
                               name='x2',
                               dtype='float32')(x)
    x3 = tf.keras.layers.Dense(7,
                               activation='softmax',
                               name='x3',
                               dtype='float32')(x)

    model = tf.keras.Model(inputs=inp, outputs=[x1, x2, x3])
    opt = tf.keras.optimizers.Adam(lr=0.00001)
    wgt = {'x1': 1.5, 'x2': 1.0, 'x3': 1.0}
    # Due to x2 factor in recall eval for grapheme root
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['categorical_accuracy'],
                  loss_weights=wgt)

    return model


from sklearn.metrics import f1_score


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, valid_data, target, fold, mm=0, rr=0, patience=10):
        self.valid_inputs = valid_data
        self.valid_outputs = target
        self.fold = fold
        self.patience = patience
        self.mm = mm
        self.rr = rr

    def on_train_begin(self, logs={}):
        self.valid_f1 = [0]
        # TODO: Grok

    def on_epoch_end(self, epoch, logs={}):

        preds = self.model.predict(self.valid_inputs)
        preds0 = np.argmax(preds[0], axis=1)
        preds1 = np.argmax(preds[1], axis=1)
        preds2 = np.argmax(preds[2], axis=1)

        r1 = recall_score(self.valid_outputs[0], preds0, average='macro')
        r2 = recall_score(self.valid_outputs[1], preds1, average='macro')
        r3 = recall_score(self.valid_outputs[2], preds2, average='macro')

        a1 = accuracy_score(self.valid_outputs[0], preds0)
        a2 = accuracy_score(self.valid_outputs[1], preds1)
        a3 = accuracy_score(self.valid_outputs[2], preds2)

        f1 = 0.5 * r1 + 0.25 * r2 + 0.25 * r3

        # LOG TO FILE
        f = open('log-%i-%i.txt' % (self.mm, self.rr), 'a')
        f.write('#' * 25)
        f.write('\n')
        f.write('#### FOLD %i EPOCH %i\n' % (self.fold + 1, epoch + 1))
        f.write('#### ACCURACY: a1=%.5f, a2=%.5f, a3=%.5f\n' % (a1, a2, a3))
        f.write('#### MACRO RECALL: r1=%.5f, r2=%.5f, r3=%.5f\n' % (r1, r2, r3))
        f.write('#### CV/LB: %.5f\n' % f1)

        print('\n')
        print('#' * 25)
        print('#### FOLD %i EPOCH %i' % (self.fold + 1, epoch + 1))
        print('#### ACCURACY: a1=%.5f, a2=%.5f, a3=%.5f' % (a1, a2, a3))
        print('#### MACRO RECALL: r1=%.5f, r2=%.5f, r3=%.5f' % (r1, r2, r3))
        print('#### CV/LB: %.5f' % f1)
        print('#' * 25)

        self.valid_f1.append(f1)
        x = np.asarray(self.valid_f1)
        if np.argsort(-x)[0] == (len(x) - self.patience - 1):
            # TODO: grok
            print('#### CV/LB no increase for %i epochs: EARLY STOPPING' % self.patience)
            f.write('#### CV/LB no increase for %i epochs: EARLY STOPPING\n' % self.patience)
            self.model.stop_training = True

        if (f1 > 0.000) & (f1 > np.max(self.valid_f1[:-1])):
            # TODO: > 0.000 ?
            print('#### Saving new best...')
            f.write('#### Saving new best...\n')
            self.model.save_weights(os.path.join(model_dir, 'fold%i-m%i-%i.h5' % (self.fold, self.mm, self.rr)))

        f.close()


# Custom learning schedule
LR_START = 1e-5
LR_MAX = 1e-3
LR_RAMPUP_EPOCHS = 5
LR_SUSTAIN_EPOCHS = 0
LR_STEP_DECAY = 0.75


def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = LR_MAX * LR_STEP_DECAY ** ((epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) // 10)
    return lr


lr2 = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

rng = [i for i in range(100)]
y = [lrfn(x) for x in rng]
plt.plot(rng, y)
plt.xlabel('epoch', size=14)
plt.ylabel('learning rate', size=14)
plt.title('Training schedule', size=16)
plt.show()

# Train model

oof1 = np.zeros((X_train.shape[0], 168))
oof2 = np.zeros((X_train.shape[0], 11))
oof3 = np.zeros((X_train.shape[0], 7))

skf = StratifiedKFold(n_splits=5,
                      shuffle=True,
                      random_state=42)

for fold, (idxT, idxV) in enumerate(skf.split(X_train, y_train[:, 0])):
    print('#' * 25)
    print('### FOLD %i' % (fold + 1))
    print('### Training on %i images. Validating on %i images' % (len(idxT), len(idxV)))
    print('#' * 25)

    K.clear_session()
    # TODO: ?
    with strategy.scope():
        model = build_model()

    train_gen = DataGenerator(X_train,
                              y_train,
                              idxT,
                              shuffle=True,
                              augment=True,
                              batch_size=BATCH_SIZE,
                              cutmix=True)
    val_x = DataGenerator(X_train,
                          y_train,
                          idxV,
                          shuffle=False,
                          augment=False,
                          cutmix=False,
                          labels=False,
                          batch_size=BATCH_SIZE * 4)
    # TODO: Grok
    val_y = [y_train[idxV, 0], y_train[idxV, 1], y_train[idxV, 2]]

    cc = CustomCallback(valid_data=val_x,
                        target=val_y,
                        fold=fold,
                        mm=mm,
                        rr=rr,
                        patience=15)
    h = model.fit(train_gen,
                  epochs=20,
                  verbose=1,
                  callbacks=[cc, lr2])

    print('#### Loading best weights...')
    model.load_weights(os.path.join(model_dir, 'fold%i-m%i-%i.h5' % (fold, mm, rr)))

    val_x = DataGenerator(X_train,
                          y_train,
                          idxV,
                          shuffle=False,
                          augment=False,
                          cutmix=False,
                          labels=False,
                          batch_size=BATCH_SIZE * 4)

    oo = model.predict(val_x)
    oof1[idxV, ] = oo[0]
    oof2[idxV, ] = oo[1]
    oof3[idxV, ] = oo[2]

    # Save OOF and IDXV   (TODO: ?)
    np.save('oof1-%i-%i' % (mm, rr), oof1)
    np.save('oof2-%i-%i' % (mm, rr), oof2)
    np.save('oof3-%i-%i' % (mm, rr), oof3)
    np.save('idxV-%i-%i' % (mm, rr), idxV)
    np.save('y_train-%i-%i' % (mm, rr), y_train)
    break

