from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmengine.model import BaseDataPreprocessor
from torch import Tensor


@MODELS.register_module()
class FrustumRangePreprocessor(BaseDataPreprocessor):
    """Frustum-Range Segmentor pre-processor for frustum region group.

    Args:
        H (int): Height of the 2D representation.
        W (int): Width of the 2D representation.
        fov_up (float): Front-of-View at upward direction of the sensor.
        fov_down (float): Front-of-View at downward direction of the sensor.
        ignore_index (int): The label index to be ignored.
        non_blocking (bool): Whether to block current process when transferring
            data to device. Defaults to False.
    """

    def __init__(self,
                 H: int,
                 W: int,
                 fov_up: float,
                 fov_down: float,
                 ignore_index: int,
                 non_blocking: bool = False) -> None:
        super(FrustumRangePreprocessor,
              self).__init__(non_blocking=non_blocking)
        self.H = H
        self.W = W
        self.fov_up = fov_up / 180 * np.pi
        self.fov_down = fov_down / 180 * np.pi
        self.fov = abs(self.fov_down) + abs(self.fov_up)
        self.ignore_index = ignore_index

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform frustum region group based on ``BaseDataPreprocessor``.

        Args:
            data (dict): Data from dataloader. The dict contains the whole
                batch data.
            training (bool): Whether to enable training time augmentation.
                Defaults to False.

        Returns:
            dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)
        data.setdefault('data_samples', None)

        inputs, data_samples = data['inputs'], data['data_samples']
        batch_inputs = dict()

        assert 'points' in inputs
        batch_inputs['points'] = inputs['points']

        voxel_dict = self.frustum_region_group(inputs['points'], data_samples)
        batch_inputs['voxels'] = voxel_dict

        return {'inputs': batch_inputs, 'data_samples': data_samples}

    @torch.no_grad()
    def frustum_region_group(self, points: List[Tensor],
                             data_samples: SampleList) -> dict:
        """Calculate frustum region of each point.

        Args:
            points (List[Tensor]): Point cloud in one data batch.

        Returns:
            dict: Frustum region information.
        """
        voxel_dict = dict()

        coors = []
        voxels = []

        for i, res in enumerate(points):
            depth = torch.linalg.norm(res[:, :3], 2, dim=1)
            yaw = -torch.atan2(res[:, 1], res[:, 0])
            pitch = torch.arcsin(res[:, 2] / depth)

            coors_x = 0.5 * (yaw / np.pi + 1.0)
            coors_y = 1.0 - (pitch + abs(self.fov_down)) / self.fov

            # scale to image size using angular resolution
            coors_x *= self.W
            coors_y *= self.H

            # round and clamp for use as index
            coors_x = torch.floor(coors_x)
            coors_x = torch.clamp(
                coors_x, min=0, max=self.W - 1).type(torch.int64)

            coors_y = torch.floor(coors_y)
            coors_y = torch.clamp(
                coors_y, min=0, max=self.H - 1).type(torch.int64)

            res_coors = torch.stack([coors_y, coors_x], dim=1)
            res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
            coors.append(res_coors)
            voxels.append(res)

            if 'pts_semantic_mask' in data_samples[i].gt_pts_seg:
                import torch_scatter
                pts_semantic_mask = data_samples[
                    i].gt_pts_seg.pts_semantic_mask
                seg_label = torch.ones(
                    (self.H, self.W),
                    dtype=torch.long,
                    device=pts_semantic_mask.device) * self.ignore_index
                res_voxel_coors, inverse_map = torch.unique(
                    res_coors, return_inverse=True, dim=0)
                voxel_semantic_mask = torch_scatter.scatter_mean(
                    F.one_hot(pts_semantic_mask).float(), inverse_map, dim=0)
                voxel_semantic_mask = torch.argmax(voxel_semantic_mask, dim=-1)
                seg_label[res_voxel_coors[:, 1],
                          res_voxel_coors[:, 2]] = voxel_semantic_mask
                data_samples[i].gt_pts_seg.semantic_seg = seg_label

        voxels = torch.cat(voxels, dim=0)
        coors = torch.cat(coors, dim=0)
        voxel_dict['voxels'] = voxels
        voxel_dict['coors'] = coors

        return voxel_dict
