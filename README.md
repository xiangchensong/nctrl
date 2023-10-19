# NCTRL

This repository contains the implementation code for paper: <br>
[**Temporally Disentangled Representation Learning
under Unknown Nonstationarity**](https://openreview.net/forum?id=V8GHCGYLkf) <br>
Xiangchen Song, Weiran Yao, Yewen Fan, Xinshuai Dong, Guangyi Chen, Juan Carlos Niebles, Eric Xing, Kun Zhang <br>
Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS 2023)
## Installation
```bash
conda create -n nctrl python=3.10 -y
conda activate nctrl
pip install torch==1.13.1 torchvision torchaudio
pip install -r requirements.txt
```

## Generate simulation data
```bash
mkdir -p data/simulation
cd tools
python generate_data.py
```
## Example usage for simulation data
```bash
python train_simulation.py -c configs/simulation/simulation_nctrl.yaml
```

## Citation
Please cite our paper if you find it useful in your research:

```
@article{song2023temporally,
  title={Temporally Disentangled Representation Learning
under Unknown Nonstationarity},
  author={Song, Xiangchen and Yao, Weiran and Fan, Yewen and Dong, Xinshuai and Chen, Guangyi and Niebles, Juan Carlos and Xing, Eric and Zhang, Kun},
  journal={NeurIPS},
  year={2023}
}
```