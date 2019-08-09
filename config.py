import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors
print_freq = 10

meta_file = 'data/LJSpeech-1.1/metadata.csv'
wave_folder = 'data/LJSpeech-1.1/wavs'

################################
# Train config                 #
################################

fp16_run = False
epochs = 100000
learning_rate = 1e-4
sigma = 1.0
iters_per_checkpoint = 2000
batch_size = 6
seed = 1234
checkpoint_path = ""
with_tensorboard = False

################################
# Data config                  #
################################

training_files = 'filelists/ljs_audio_text_train_filelist.txt'
validation_files = 'filelists/ljs_audio_text_val_filelist.txt'
segment_length = 16000
sampling_rate = 22050
filter_length = 1024
hop_length = 256
win_length = 1024
mel_fmin = 0.0
mel_fmax = 8000.0

################################
# Waveglow config              #
################################

n_mel_channels = 80
n_flows = 12
n_group = 8
n_early_every = 4
n_early_size = 2
# WN_config
n_layers = 8
n_channels = 256
kernel_size = 3
