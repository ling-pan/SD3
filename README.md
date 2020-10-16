# Softmax Deep Double Deterministic Policy Gradients

This repository is the implementation of Softmax Deep Deterministic Policy Gradients in NeurIPS 2020, and is based on the open-source [TD3](https://github.com/sfujim/TD3) codebase.

Please cite our paper if you use this codebase:

```
@inproceedings{pan2020softmax,
  title={Softmax Deep Double Deterministic Policy Gradients},
  author={Pan, Ling and Cai, Qingpeng and Huang, Longbo},
  booktitle={Neural Information Processing System},
  year={2020}
}
```

## Requirements
- python: 3.5.2
- mujoco_py: 1.50.1.68
- torch: 1.3.0
- gym: 0.15.3
- box2d-py

## Usage
To replicate the results in the paper:
```
./run.sh
```
To run the SD3 algorithm in each single environment:

```
python main.py --env <environment_name>
```