# WaveGlow

A PyTorch implementation of WaveGlow, described in [WaveGlow: A Flow-based Generative Network for Speech Synthesis](https://arxiv.org/abs/1811.00002), a flow-based network capable of generating high quality speech from mel-spectrograms.

## Dataset

[LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/)

## Dependency

- Python 3.5.2
- PyTorch 1.0.0

## Usage
### Data Pre-processing
Extract data:
```bash
$ python extract.py
```

### Train
```bash
$ python train.py
```

If you want to visualize during training, run in your terminal:
```bash
$ tensorboard --logdir runs
```

### Demo
Pick 10 random test examples from test set:
```bash
$ python demo.py