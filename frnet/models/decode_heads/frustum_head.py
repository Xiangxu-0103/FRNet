from typing import List

import torch
import torch.nn as nn
from mmdet3d.models import Base3DDecodeHead
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import ConfigType, OptConfigType
from torch import Tensor


@MODELS.register_module()
class FrustumHead(Base3DDecodeHead):

    def __init__(self,
                 loss_ce: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=False,
                     class_weight=None,
                     loss_weight=1.0),
                 loss_dice: OptConfigType = None,
                 loss_lovasz: OptConfigType = None,
                 loss_boundary: OptConfigType = None,
                 indices: int = 0,
                 **kwargs) -> None:
        super(FrustumHead, self).__init__(**kwargs)

        self.loss_ce = MODELS.build(loss_ce)
        if loss_dice is not None:
            self.loss_dice = MODELS.build(loss_dice)
        else:
            self.loss_dice = None
        if loss_lovasz is not None:
            self.loss_lovasz = MODELS.build(loss_lovasz)
        else:
            self.loss_lovasz = None
        if loss_boundary is not None:
            self.loss_boundary = MODELS.build(loss_boundary)
        else:
            self.loss_boundary = None

        self.indices = indices

    def build_conv_seg(self, channels: int, num_classes: int,
                       kernel_size: int) -> nn.Module:
        return nn.Conv2d(
            channels,
            num_classes,
            kernel_size=kernel_size,
            padding=kernel_size // 2)

    def forward(self, voxel_dict: dict) -> dict:
        """Forward function."""
        seg_logit = self.cls_seg(voxel_dict['voxel_feats'][self.indices])
        voxel_dict['seg_logit'] = seg_logit
        return voxel_dict

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_pts_seg.semantic_seg
            for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)

    def loss_by_feat(self, voxel_dict: dict,
                     batch_data_samples: SampleList) -> dict:
        seg_label = self._stack_batch_gt(batch_data_samples)
        seg_logit = voxel_dict['seg_logit']

        loss = dict()
        loss['loss_ce'] = self.loss_ce(
            seg_logit, seg_label, ignore_index=self.ignore_index)
        if self.loss_dice:
            loss['loss_dice'] = self.loss_dice(
                seg_logit, seg_label, ignore_index=self.ignore_index)
        if self.loss_lovasz:
            loss['loss_lovasz'] = self.loss_lovasz(
                seg_logit, seg_label, ignore_index=self.ignore_index)
        if self.loss_boundary:
            loss['loss_boundary'] = self.loss_boundary(seg_logit, seg_label)
        return loss

    def predict(self, voxel_dict: dict, batch_input_metas: List[dict],
                test_cfg: ConfigType) -> List[Tensor]:
        voxel_dict = self.forward(voxel_dict)

        seg_pred_list = self.predict_by_feat(voxel_dict, batch_input_metas)

        final_seg_pred_list = []
        for seg_pred, input_metas in zip(seg_pred_list, batch_input_metas):
            if 'num_points' in input_metas:
                num_points = input_metas.num_points
                seg_pred = seg_pred[:num_points]
            final_seg_pred_list.append(seg_pred)
        return final_seg_pred_list

    def predict_by_feat(self, voxel_dict: dict,
                        batch_input_metas: List[dict]) -> List[Tensor]:
        seg_logits = voxel_dict['seg_logit']
        seg_logits = seg_logits.permute(0, 2, 3, 1).contiguous()

        coors = voxel_dict['coors']
        seg_pred_list = []
        for batch_idx in range(len(batch_input_metas)):
            batch_mask = coors[:, 0] == batch_idx
            res_coors = coors[batch_mask]
            proj_x = res_coors[:, 2]
            proj_y = res_coors[:, 1]
            seg_pred_list.append(seg_logits[batch_idx, proj_y, proj_x])
        return seg_pred_list
