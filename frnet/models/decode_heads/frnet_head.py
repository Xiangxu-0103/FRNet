from typing import List, Sequence

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmdet3d.models import Base3DDecodeHead
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import ConfigType
from torch import Tensor


@MODELS.register_module()
class FRHead(Base3DDecodeHead):

    def __init__(self,
                 in_channels: int,
                 middle_channels: Sequence[int],
                 norm_cfg: ConfigType = dict(type='BN1d'),
                 loss_ce: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=False,
                     class_weight=None,
                     loss_weight=1.0),
                 **kwargs) -> None:
        super(FRHead, self).__init__(**kwargs)

        self.loss_ce = MODELS.build(loss_ce)

        self.mlps = nn.ModuleList()
        for i in range(len(middle_channels)):
            out_channels = middle_channels[i]
            self.mlps.append(
                nn.Sequential(
                    nn.Linear(in_channels, out_channels, bias=False),
                    build_norm_layer(norm_cfg, out_channels)[1],
                    nn.ReLU(inplace=True)))
            in_channels = out_channels

    def build_conv_seg(self, channels: int, num_classes: int,
                       kernel_size: int) -> nn.Module:
        return nn.Linear(channels, num_classes)

    def forward(self, voxel_dict: dict) -> dict:
        point_feats_backbone = voxel_dict['point_feats_backbone'][0]
        point_feats = voxel_dict['point_feats'][:-1]
        voxel_feats = voxel_dict['voxel_feats'][0]
        voxel_feats = voxel_feats.permute(0, 2, 3, 1)
        pts_coors = voxel_dict['coors']
        map_point_feats = voxel_feats[pts_coors[:, 0], pts_coors[:, 1],
                                      pts_coors[:, 2]]

        for i, mlp in enumerate(self.mlps):
            map_point_feats = mlp(map_point_feats)
            if i == 0:
                map_point_feats = map_point_feats + point_feats_backbone
            else:
                map_point_feats = map_point_feats + point_feats[-i]
        seg_logit = self.cls_seg(map_point_feats)
        voxel_dict['seg_logit'] = seg_logit
        return voxel_dict

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_pts_seg.pts_semantic_mask
            for data_sample in batch_data_samples
        ]
        return torch.cat(gt_semantic_segs, dim=0)

    def loss_by_feat(self, voxel_dict: dict,
                     batch_data_samples: SampleList) -> dict:
        seg_logit = voxel_dict['seg_logit']
        seg_label = self._stack_batch_gt(batch_data_samples)

        loss = dict()
        loss['loss_ce'] = self.loss_ce(
            seg_logit, seg_label, ignore_index=self.ignore_index)
        return loss

    def predict(self, voxel_dict: dict, batch_input_metas: List[dict],
                test_cfg: ConfigType) -> List[Tensor]:
        voxel_dict = self.forward(voxel_dict)

        seg_pred_list = self.predict_by_feat(voxel_dict, batch_input_metas)

        final_seg_pred_list = []
        for seg_pred, input_metas in zip(seg_pred_list, batch_input_metas):
            if 'num_points' in input_metas:
                num_points = input_metas['num_points']
                seg_pred = seg_pred[:num_points]
            final_seg_pred_list.append(seg_pred)
        return final_seg_pred_list

    def predict_by_feat(self, voxel_dict: dict,
                        batch_input_metas: List[dict]) -> List[Tensor]:
        seg_logits = voxel_dict['seg_logit']

        coors = voxel_dict['coors']
        seg_pred_list = []
        for batch_idx in range(len(batch_input_metas)):
            batch_mask = coors[:, 0] == batch_idx
            seg_pred_list.append(seg_logits[batch_mask])
        return seg_pred_list
