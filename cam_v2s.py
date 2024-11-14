# Original CAM Code is modified from Yang et al. ICASSP 2021 (https://arxiv.org/pdf/2010.13309.pdf)
# Please consider to cite both de Andrade et al. 2018 and Yang et al. 2021 ICML, if you use the attention heads and CAM visualization.

from ts_model import  AttRNN_Model, ARTLayer, WARTmodel
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import ZeroPadding2D, Input, Layer, ZeroPadding1D
import matplotlib.pyplot as plt
import time as ti
from utils import layer_output, to_rgb
import librosa
import librosa.display
from ts_dataloader import readucr
import argparse
from PIL import Image
data_ix = ti.strftime("%m%d_%H%M")

base_model = AttRNN_Model()
base_model.summary()

audios = np.load("Datasets/val_audios.npy") # load wavs files
cmds = np.load("Datasets/val_cmds.npy")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=int, default=5, help="Ford-A (0), Beef (1), ECG200 (2), Wine (3)") 
parser.add_argument("--weight", type=str, default="beta/No0_map1-epoch10-val_acc0.9371", help="weight in /weights/")
parser.add_argument("--mapping", type=int, default=6, help="number of multi-mapping")
parser.add_argument("--seg", type=int, default=3, help="the # of segments")
parser.add_argument("--segplace", type=str, default="start", choices=['start', 'center', 'end'], help="Placement within segments: 'start', 'center', or 'end'")
parser.add_argument("--segstack", action='store_true', help="Whether to stack target segments consecutively")
parser.add_argument("--layer", type=str, default="conv2d_1", help="the layer for cam")
args = parser.parse_args()


idAudio = 0
GSCmdV2Categs = {
            'unknown': 0,
            'silence': 0,
            '_unknown_': 0,
            '_silence_': 0,
            '_background_noise_': 0,
            'yes': 2,
            'no': 3,
            'up': 4,
            'down': 5,
            'left': 6,
            'right': 7,
            'on': 8,
            'off': 9,
            'stop': 10,
            'go': 11,
            'zero': 12,
            'one': 13,
            'two': 14,
            'three': 15,
            'four': 16,
            'five': 17,
            'six': 18,
            'seven': 19,
            'eight': 20,
            'nine': 1,
            'backward': 21,
            'bed': 22,
            'bird': 23,
            'cat': 24,
            'dog': 25,
            'follow': 26,
            'forward': 27,
            'happy': 28,
            'house': 29,
            'learn': 30,
            'marvin': 31,
            'sheila': 32,
            'tree': 33,
            'visual': 34,
            'wow': 35}


key_list = list(GSCmdV2Categs.keys())
cmd_k = key_list[cmds[idAudio]]
print("Input Speech Cmd: ", cmd_k)


model = base_model

attM = Model(inputs=model.input, outputs=[model.get_layer('output').output, 
                                          model.get_layer('attSoftmax').output,
                                          model.get_layer('mel_stft').output])


x_train, y_train, x_test, y_test = readucr(args.dataset) # 4 - Earthquake // 8 - ECG 5k
tmp_xt = x_test
classes = np.unique(np.concatenate((y_train, y_test), axis=0))
num_classes = len(np.unique(y_train))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
target_shape = x_test[0].shape

art_model = WARTmodel(target_shape, model, 36,  args.mapping, num_classes, mod=2)

checkpoint_path = "weight/" + args.weight
checkpoint = tf.train.Checkpoint(model=art_model)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()

ReproM = Model(inputs=art_model.input, outputs=[art_model.get_layer('reshape_1').output])

repros = ReproM.predict(x_test)


def visual_sp(use='base', clayer=args.layer):
    # Check if the specified layer exists in the base_model
    layer_names = [layer.name for layer in base_model.layers]
    if clayer not in layer_names:
        print(f"Error: Specified layer '{clayer}' not found in base_model.")
        print("Available layers:", layer_names)
        return  # Exit the function if layer is not found

    print(f"Using layer '{clayer}' for CAM visualization.")

    # Define an intermediate model to get the output from the input layer after reprogramming
    reprogram_layer = art_model.get_layer('reshape_1').input
    reprogram_model = Model(inputs=art_model.input, outputs=reprogram_layer)

    # Get the reprogrammed input (xt') for visualization based on x_test
    reprogrammed_input = reprogram_model.predict(x_test)

    # Plot the reprogrammed input time series
    plt.figure()
    plt.style.use("seaborn-whitegrid")
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 20))

    # Plot the reprogrammed input time series for a specific sample
    ax1.set_ylabel('Amplitude', fontsize=18)
    ax1.set_xlabel('Sample index', fontsize=18)
    ax1.plot(reprogrammed_input[idAudio], 'b-', label="Reprogrammed time series")
    
    if use != 'base':
        # Use SegZeroPadding1D to visualize the target placement within reprogrammed input
        x_tmp = x_test[idAudio].reshape((x_test[idAudio].shape[0], 1))  
        x_tmp = tf.expand_dims(x_tmp, axis=0)
        aug_tmp = SegZeroPadding1D(x_tmp, int(args.seg), x_test[idAudio].shape[0], args.segplace, args.segstack)
        ax1.plot(tf.squeeze(tf.squeeze(aug_tmp, axis=0), axis=-1), 'k-', label="Target time series")
    ax1.legend(fancybox=True, framealpha=1, borderpad=1, fontsize=16)

    # Remove the last dimension from x_test to match the expected input shape for attM
    x_test_2d = x_test.squeeze(-1)

    # Predict with attM to get attention weights and spectrograms for visualization
    outs, attW, specs = attM.predict(x_test_2d)
    
    # Define w_x and h_x based on the shape of x_test instead of specs
    w_x, h_x = x_test[idAudio].shape[0], specs.shape[2]  # assuming the second dimension of specs is the frequency dimension

    # Log of attention weights
    ax2.set_ylabel('Log of attention weight', fontsize=18)
    ax2.set_xlabel('Mel-spectrogram index', fontsize=18)
    ax2.plot(np.log(attW[idAudio]), 'r-')

    # Spectrogram visualization
    img3 = ax3.pcolormesh(specs[idAudio, :, :, 0])
    ax3.set_ylabel('Frequency', fontsize=18)
    ax3.set_xlabel('Time', fontsize=18)

    # Class Activation Mapping visualization
    i_heatmap1, _ = layer_output(x_test, base_model, clayer, idAudio)
    i_cam1 = to_rgb(i_heatmap1, w_x, h_x)  # Convert heatmap to RGB based on x_test dimensions
    img4 = ax4.imshow(i_cam1, aspect="auto")
    ax4.set_xticks([])
    ax4.set_yticks([])

    plt.tight_layout()
    if use == 'base':
        plt.savefig("results/" + data_ix + "_sp.png")
    else:
        plt.savefig("results/" + data_ix + "_ts_No"+ str(args.dataset) +".png")




def SegZeroPadding1D(orig_x, seg_num, orig_xlen, place='start', stack=False):
    src_xlen = 16000  # Total length of the reprogrammed input
    aug_x = tf.zeros([src_xlen, 1])  # Initialize the reprogrammed input with zeros

    if stack:
        # Calculate the consecutive start positions based on `place`
        for s in range(seg_num):
            if place == 'start':
                startidx = s * orig_xlen
            elif place == 'center':
                startidx = s * orig_xlen + (src_xlen - seg_num * orig_xlen) // 2
            elif place == 'end':
                startidx = src_xlen - (seg_num - s) * orig_xlen
            else:
                raise ValueError("Invalid value for 'place'. Choose from 'start', 'center', or 'end'.")

            endidx = startidx + orig_xlen
            if endidx > src_xlen:
                break  # Avoid out-of-bounds placement

            # Place each stacked segment with target data at calculated position
            seg_x = ZeroPadding1D(padding=(startidx, src_xlen - endidx))(orig_x)
            aug_x += seg_x

    else:
        # Non-stacking case: evenly space segments within the total length
        segment_length = src_xlen // seg_num
        # Determine offset based on `place`
        if place == 'start':
            offset = 0
        elif place == 'center':
            offset = (segment_length - orig_xlen) // 2
        elif place == 'end':
            offset = segment_length - orig_xlen
        else:
            raise ValueError("Invalid value for 'place'. Choose from 'start', 'center', or 'end'.")

        for s in range(seg_num):
            startidx = s * segment_length + offset
            endidx = startidx + orig_xlen
            if endidx > src_xlen:
                break  # Avoid out-of-bounds placement

            # Place target data at the specified location within each segment
            seg_x = ZeroPadding1D(padding=(startidx, src_xlen - endidx))(orig_x)
            aug_x += seg_x

    return aug_x




visual_sp(use="adv")

