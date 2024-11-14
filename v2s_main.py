# Apache Apache-2.0 License
from tensorflow.keras.layers import Dense, ZeroPadding1D, Reshape
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np
## time series NN models
from ts_model import AttRNN_Model, ARTLayer, WARTmodel, make_model, VGGish_Model
from ts_dataloader import readucr, plot_acc_loss
import argparse

# List available devices
physical_devices = tf.config.list_physical_devices()
print("Available physical devices:")
for device in physical_devices:
    print(device)


# Learning phase is set to 0 since we want the network to use the pretrained moving mean/var
K.clear_session()
# K.set_learning_phase(0)

parser = argparse.ArgumentParser()
# parser.add_argument("--mod", type = int, default = 2, help = "Single input seq (0), multiple input aug (1), repro w/ TF (2)")
# parser.add_argument("--net", type = int, default = 0, help = "Pretrained (0), AttRNN (#32), (1) VGGish (#512)")
# parser.add_argument("--dataset", type = int, default = 0, help = "Ford-A (0), Beef (1), ECG200 (2), Wine (3), Earthquakes (4), Worms (5), Distal (6), Outline Correct (7), ECG-5k (8), ArrowH (9), CBF (10), ChlorineCon (11)")
# parser.add_argument("--mapping", type= int, default=1, help = "number of multi-mapping")
# parser.add_argument("--eps", type = int, default = 100, help = "Epochs") 
# parser.add_argument("--per", type = int, default = 0, help = "save weight per N epochs")
# parser.add_argument("--dr", type=int, default = 4, help = "drop out rate")
parser.add_argument("--seg", type=int, default = 1, help = "seg padding number")
args = parser.parse_args()

mod = 2
net = 1
dataset = 2
mapping = 18
eps = 40
per = 0
dr = 4
seg = args.seg
stack = False
place = "center"


x_train, y_train, x_test, y_test = readucr(dataset)
    
y_train = [np.uint32(i) for i in y_train]
y_test = [np.uint32(i) for i in y_test]

classes = np.unique(np.concatenate((y_train, y_test), axis=0))

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# The x_test and y_test are the official validation set in the UCR.

num_classes = len(np.unique(y_train))

if np.min(np.unique(y_train))!=0: #for those tasks with labels not start from 0, shift to 0 
    max_v = np.max(y_train)
    y_train=[i%max_v  for i in y_train]
    y_test=[i%max_v  for i in y_test]

idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = np.array(y_train)[idx]

print("--- X shape : ", x_train[0].shape, "--- Num of Classes : ", num_classes) ## target class


## Pre-trained Model for Adv Program  
if net == 0:
    pr_model = AttRNN_Model()
elif net == 1: # fine-tuning with additive dense layer
    pr_model = VGGish_Model()
elif net == 2: # audio-set output classes  = 128
    pr_model = VGGish_Model(audioset = True)
elif net == 3: # unet
    pr_model = AttRNN_Model(unet= True)


# pr_model.summary()

## # of Source classes in Pre-trained Model
if net != 2: ## choose pre-trained network 
    source_classes = 36 ## Google Speech Commands
elif net == 2:
    source_classes = 128 ## AudioSet by VGGish
else:
    source_classes = 512 ## VGGish feature num

target_shape = x_train[0].shape

## Adv Program Time Series (ART)
mapping_num = mapping
seg_num = seg
drop_rate = dr*0.1

#pr_model.summary()



try:
    assert mapping_num*num_classes <= source_classes
except AssertionError:
    print("Error: The mapping num should be smaller than source_classes / num_classes: {}".format(source_classes//num_classes)) 
    exit(1)

model = WARTmodel(target_shape, pr_model, source_classes, mapping_num, num_classes, mod, seg_num, drop_rate, place, stack)
# else:
# model = pr_model # define for transfer learning


## Loss
adam = tf.keras.optimizers.Adam(lr=0.05,decay=0.48)
save_path = "weight/beta/No{}_map{}".format(dataset, mapping)
if per!= 0:
    checkpoints = tf.keras.callbacks.ModelCheckpoint(
    filepath=save_path + "-epoch{epoch:02d}-val_acc{val_accuracy:.4f}", 
    save_weights_only=True, 
    save_freq='epoch', 
    save_format='tf',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
    )
    exp_callback = [tf.keras.callbacks.EarlyStopping(patience=500), checkpoints]
else:
    exp_callback = [tf.keras.callbacks.EarlyStopping(patience=500)]


model.compile(loss='categorical_crossentropy', optimizer = adam, metrics=['accuracy'])

model.summary()

batch_size = 32
epochs = eps

# convert class vectors to binary class matrices
if mod == 0: # single input w/ random mapping
    y_train = keras.utils.to_categorical(y_train, source_classes)
    y_test = keras.utils.to_categorical(y_test, source_classes)
else: # with many to one mapping
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)


exp_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
          validation_data=(x_test,y_test), callbacks= exp_callback)

score = model.evaluate(x_train, y_train, verbose=0)
print('--- Train loss:', score[0])
print('- Train accuracy:', score[1])

score = model.evaluate(x_test, y_test, verbose=0)
print('--- Test loss:', score[0])
print('- Test accuracy:', score[1])


print("=== Best Val. Acc: ", max(exp_history.history['val_accuracy']), " At Epoch of ", np.argmax(exp_history.history['val_accuracy']) + 1)

plot_acc_loss(exp_history, str(eps), str(dataset), str(mapping), str(seg))


