# Installation

## Prerequisties

This codebase is tested with `torch==1.8.1`, `mmengine==0.9.0`, `mmcv==2.1.0`, `mmdet==3.2.0`, and `mmdet3d==1.3.0`, with `CUDA 10.2` and `CUDA 11.1`.

**Step 1**. Create a conda environment and activate it.

```bash
conda create --name frnet python==3.8 -y
conda activate frnet
```

**Step 2**. Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/)

```bash
conda install pytorch torchvision -c pytorch
```

**Step 3**. Install [MMEngine](https://github.com/open-mmlab/mmengine), [MMCV](https://github.com/open-mmlab/mmcv), [MMDetection](https://github.com/open-mmlab/mmdetection), [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) using [MIM](https://github.com/open-mmlab/mim).

```bash
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet
mim install mmdet3d
```

Optionally, You can also install the above projects from the source, e.g.:

```bash
git clone https://github.com/open-mmlab/mmdetection3d
cd mmdetection3d
pip install -v -e .
```

Meanwhile, you also need to install [`torch-scatter`](https://github.com/rusty1s/pytorch_scatter) and [`nuScenes devkit`](https://github.com/nutonomy/nuscenes-devkit).
