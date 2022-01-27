import matplotlib
import matplotlib.pylab as plt

import data_utils
from numpy.core.defchararray import asarray
import librosa
import IPython.display as ipd
import os
import sys
import numpy as np
import torch
import IPython
from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence


def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom', 
                       interpolation='none')

hparams = create_hparams()
hparams.sampling_rate = 22050

checkpoint_path = "/home/tuht/tacotron2/outdir/checkpoint_33000"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()

text = "Ai đây tức là một kẻ ăn mày vậy. Anh ta chưa kịp quay đi thì đã thấy mấy con chó vàng chạy xồng xộc ra cứ nhảy xổ vào chân anh."
sequence = np.array(text_to_sequence(text, ['basic_cleaners']))[None, :]
sequence = torch.autograd.Variable(
    torch.from_numpy(sequence)).cuda().long()

mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)

mel_outputs_postnet =  mel_outputs_postnet.detach().cpu().numpy()

np.save(os.path.join("/home/tuht/hifi-gan/test_mel_files", "testfile.npy"), mel_outputs_postnet, allow_pickle=False)
print(mel_outputs_postnet.shape)
print(mel_outputs_postnet)
# mel_outputs_postnet = mel_outputs_postnet.reshape(-1,80)

# S = librosa.feature.inverse.mel_to_stft(mel_outputs_postnet).astype("int16")
# y = librosa.griffinlim(S)
