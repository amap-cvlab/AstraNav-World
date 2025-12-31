# AstraNav-World: World Model for Foresight Control and Consistency

[![Home Page](https://img.shields.io/badge/Project-AstraNavWorld-blue.svg)](https://astra-amap.github.io/AstraNav-World.github.io/)
[![arXiv](https://img.shields.io/badge/Arxiv-2512.21714-b31b1b.svg?logo=arXiv)](https://arxiv.org/pdf/2512.21714)

## ✨ Core Highlights
* 🧠 VLM-Wan Symbiosis: Combines Qwen2.5-VL reasoning with Wan2.2 physical priors to enable robust long-horizon world modeling.
* 🔄 Bidirectional Consistency: Jointly couples vision and action to eliminate cumulative errors and "physical hallucinations."
* 🌍 Zero-Shot Sim-to-Real: Direct deployment to physical robots without any real-world fine-tuning, powered by generalized spatial understanding.

### Architecture
<img src="assets/Architecture.png" width="600">

### Consistency
<img src="assets/consis.png" width="600">

## 🔥 Latest News!!
* December 29, 2025: We release the inference code of action former for the R2R, RxR, and OVON benchmarks. 

## Coming Soon
* vlm, wan, policy: ckpt
* training of action former
* inference of diffusion policy
* training of diffusion policy

## Quickstart
### 🧰 Installation

Clone the repo:

```
git clone https://github.com/amap-cvlab/AstraNav-World.git
```

Install Training dependencies:
```
# Ensure torch >= 2.6.0
pip install -r requirements.txt
cd wan 
pip install .
```

Install habitat-sim and habitat-lab for inference
```
r2r&rxr and ovon have different version dependencies.
● habitat-sim
r2r&rxr: git clone https://github.com/facebookresearch/habitat-sim.git && cd habitat-sim && git checkout v0.1.7 
Ovon: git clone https://github.com/facebookresearch/habitat-sim.git && cd habitat-sim && git checkout v0.2.3
pip install -r requirements.txt
python setup.py install --headless
● habitat-lab
r2r&rxr: git clone https://github.com/facebookresearch/habitat-lab && cd habitat-lab && git checkout v0.1.7
ovon: git clone https://github.com/chongchong2025/habitat-lab && cd habitat-lab && git checkout v0.2.3_waypoint
python -m pip install -r habitat-baselines/habitat_baselines/rl/requirements.txt
python -m pip install -r habitat-baselines/habitat_baselines/rl/ddppo/requirements.txt
pip install -e .
cd habitat-baselines
pip install -e .
```

### ⚡ Inference
● r2r & rxr
``` sh
cd infer_r2r_rxr
bash eval_r2r.sh
bash eval_rxr.sh
```
● ovon
``` sh
cd infer_ovon
bash eval_ovon.sh
```

## 🏛️ Citation
If you find this repository useful, please consider giving a star ⭐ and citation
```
@misc{hu2025astranavworldworldmodelforesight,
      title={AstraNav-World: World Model for Foresight Control and Consistency}, 
      author={Junjun Hu and Jintao Chen and Haochen Bai and Minghua Luo and Shichao Xie and Ziyi Chen and Fei Liu and Zedong Chu and Xinda Xue and Botao Ren and Xiaolong Wu and Mu Xu and Shanghang Zhang},
      year={2025},
      eprint={2512.21714},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.21714}, 
}
@article{xue2025omninav,
  title={OmniNav: A Unified Framework for Prospective Exploration and Visual-Language Navigation},
  author={Xue, Xinda and Hu, Junjun and Luo, Minghua and Shichao, Xie and Chen, Jintao and Xie, Zixun and Kuichen, Quan and Wei, Guo and Xu, Mu and Chu, Zedong},
  journal={arXiv preprint arXiv:2509.25687},
  year={2025}
}
```

## Acknowledgments
Thanks to [OmniNav](https://github.com/amap-cvlab/OmniNav), [Navid](https://github.com/jzhzhang/NaVid-VLN-CE), [MTU3D](https://github.com/MTU3D/MTU3D), and [Ovon](https://github.com/naokiyokoyama/ovon) for open-sourcing the construction of training data and the closed-loop inference code. Their contributions have significantly enriched the open-source community.

