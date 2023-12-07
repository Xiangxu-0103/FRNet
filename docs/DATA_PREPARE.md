# Data Preparation

## Overall Structure

```
FRNet
├── data
│   ├── nuscenes
│   │   ├── lidarseg
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   ├── semantickitti
│   │   ├── sequences
│   │   │   ├── 00
│   │   │   │   ├── labels
│   │   │   │   ├── velodyne
│   │   │   ├── 01
│   │   │   ├── ..
│   │   │   ├── 21
```

## nuScenes

We need to create `.pkl` info files for nuScenes following the instruction of MMDetection3D. Meanwhile, we also support a simplified version for lidarseg only by running:

```bash
python tools/create_nuscenes.py --root-path ${PATH_TO_NUSCENES}
```

## SemanticKITTI

We need to create `.pkl` info files for SemanticKITTI following the instruction of MMDetection3D. Meanwhile, we also support a simplified version by running:

```bash
python tools/create_semantickitti.py --root-path ${PATH_TO_SEMANTICKITTI}
```
