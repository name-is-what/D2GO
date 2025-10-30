# [MM 2025] Test-time Graph OOD Detection via Dynamic Dictionary Expansion and OOD Score Calibration
This repository is the official implementation of "[Test-time Graph OOD Detection via Dynamic Dictionary Expansion and OOD Score Calibration](https://dl.acm.org/doi/abs/10.1145/3746027.3755741)" accepted by the 33rd ACM International Conference on Multimedia (MM-2025).

[![Black Logo](frame.png)](https://dl.acm.org/doi/abs/10.1145/3746027.3755741)

------

## 0. Abstract



## 1. Requirements

Main package requirements:

- `CUDA == 11.1`
- `Python == 3.7.12`
- `PyTorch == 1.8.0`
- `PyTorch-Geometric == 2.0.0`

To install the complete requiring packages, use the following command at the root directory of the repository:

```setup
pip install -r requirements.txt
```

## 2. Quick Start
Just run the script corresponding to the experiment and dataset you want. For instance:

* Run out-of-distribution detection on all dataset pairs:
```bash
bash run.sh
```


## 3. Citation
If you find this repository helpful, please consider citing the following paper. We welcome any discussions with [hou_yue@buaa.edu.cn](mailto:hou_yue@buaa.edu.cn).

```bibtex
inproceedings{hou2025test,
  title={Test-time Graph OOD Detection via Dynamic Dictionary Expansion and OOD Score Calibration},
  author={Hou, Yue and Su, Yingke and Wu, Junran and Xu, Ke},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={8845--8853},
  year={2025}
}
```
