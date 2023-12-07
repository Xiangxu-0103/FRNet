from typing import List, Optional, Sequence

import numpy as np
import torch
from mmcv.transforms import BaseTransform, Compose
from mmdet3d.registry import TRANSFORMS
from mmengine.utils import is_list_of


@TRANSFORMS.register_module()
class FrustumMix(BaseTransform):

    def __init__(self,
                 H: int,
                 W: int,
                 fov_up: float,
                 fov_down: float,
                 num_areas: List[int],
                 pre_transform: Optional[Sequence[dict]] = None,
                 prob: float = 1.0) -> None:
        assert is_list_of(num_areas, int)
        self.num_areas = num_areas

        self.H = H
        self.W = W
        self.fov_up = fov_up / 180 * np.pi
        self.fov_down = fov_down / 180 * np.pi
        self.fov = abs(self.fov_down) + abs(self.fov_up)

        self.prob = prob
        if pre_transform is None:
            self.pre_transform = pre_transform
        else:
            self.pre_transform = Compose(pre_transform)

    def frustum_vertical_mix_transform(self, input_dict: dict,
                                       mix_results: dict) -> dict:
        points = input_dict['points']
        pts_semantic_mask = input_dict['pts_semantic_mask']

        mix_points = mix_results['points']
        mix_pts_semantic_mask = mix_results['pts_semantic_mask']

        depth = torch.linalg.norm(points.coord[:, :3], 2, dim=1)
        pitch = torch.arcsin(points.coord[:, 2] / depth)
        coors = 1.0 - (pitch + abs(self.fov_down)) / self.fov
        coors *= self.H
        coors = torch.floor(coors)
        coors = torch.clamp(coors, min=0, max=self.H - 1).type(torch.int64)

        mix_depth = torch.linalg.norm(mix_points.coord[:, :3], 2, dim=1)
        mix_pitch = torch.arcsin(mix_points.coord[:, 2] / mix_depth)
        mix_coors = 1.0 - (mix_pitch + abs(self.fov_down)) / self.fov
        mix_coors *= self.H
        mix_coors = torch.floor(mix_coors)
        mix_coors = torch.clamp(
            mix_coors, min=0, max=self.H - 1).type(torch.int64)

        num_areas = np.random.choice(self.num_areas, size=1)[0]
        row_list = np.linspace(0, self.H, num_areas + 1, dtype=np.int64)
        out_points = []
        out_pts_semantic_mask = []
        for i in range(num_areas):
            start_row = row_list[i]
            end_row = row_list[i + 1]
            if i % 2 == 0:
                idx = (coors >= start_row) & (coors < end_row)
                out_points.append(points[idx])
                out_pts_semantic_mask.append(pts_semantic_mask[idx.numpy()])
            else:
                idx = (mix_coors >= start_row) & (mix_coors < end_row)
                out_points.append(mix_points[idx])
                out_pts_semantic_mask.append(
                    mix_pts_semantic_mask[idx.numpy()])
        out_points = points.cat(out_points)
        out_pts_semantic_mask = np.concatenate(out_pts_semantic_mask, axis=0)
        input_dict['points'] = out_points
        input_dict['pts_semantic_mask'] = out_pts_semantic_mask
        return input_dict

    def frustum_horizontal_mix_transform(self, input_dict: dict,
                                         mix_results: dict) -> dict:
        points = input_dict['points']
        pts_semantic_mask = input_dict['pts_semantic_mask']

        mix_points = mix_results['points']
        mix_pts_semantic_mask = mix_results['pts_semantic_mask']

        yaw = -torch.atan2(points.coord[:, 1], points.coord[:, 0])
        coors = 0.5 * (yaw / np.pi + 1.0)
        coors *= self.W
        coors = torch.floor(coors)
        coors = torch.clamp(coors, min=0, max=self.W - 1).type(torch.int64)

        mix_yaw = -torch.atan2(mix_points.coord[:, 1], mix_points.coord[:, 0])
        mix_coors = 0.5 * (mix_yaw / np.pi + 1.0)
        mix_coors *= self.W
        mix_coors = torch.floor(mix_coors)
        mix_coors = torch.clamp(
            mix_coors, min=0, max=self.W - 1).type(torch.int64)

        start_col = np.random.randint(0, self.W // 2)
        end_col = start_col + self.W // 2

        idx = (coors < start_col) | (coors >= end_col)
        mix_idx = (mix_coors >= start_col) & (mix_coors < end_col)

        out_points = points.cat([points[idx], mix_points[mix_idx]])
        out_pts_semantic_mask = np.concatenate(
            (pts_semantic_mask[idx.numpy()],
             mix_pts_semantic_mask[mix_idx.numpy()]),
            axis=0)
        input_dict['points'] = out_points
        input_dict['pts_semantic_mask'] = out_pts_semantic_mask
        return input_dict

    def transform(self, input_dict: dict) -> dict:
        if np.random.rand() > self.prob:
            return input_dict

        assert 'dataset' in input_dict, \
            '`dataset` is needed to pass through FrustumMix, while not found.'
        dataset = input_dict['dataset']

        # get index of other point cloud
        index = np.random.randint(0, len(dataset))

        mix_results = dataset.get_data_info(index)

        if self.pre_transform is not None:
            # pre_transform may also require dataset
            mix_results.update({'dataset': dataset})
            # before frustummix need to go through the necessary pre_transform
            mix_results = self.pre_transform(mix_results)
            mix_results.pop('dataset')

        if np.random.rand() > 0.5:
            # frustummix along vertical direction
            input_dict = self.frustum_vertical_mix_transform(
                input_dict, mix_results)
        else:
            # frustummix along horizontal direction
            input_dict = self.frustum_horizontal_mix_transform(
                input_dict, mix_results)
        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(H={self.H}, '
        repr_str += f'W={self.W}, '
        repr_str += f'fov_up={self.fov_up}, '
        repr_str += f'fov_down={self.fov_down}, '
        repr_str += f'num_areas={self.num_areas}, '
        repr_str += f'pre_transform={self.pre_transform}, '
        repr_str += f'prob={self.prob})'
        return repr_str


@TRANSFORMS.register_module()
class RangeInterpolation(BaseTransform):

    def __init__(self,
                 H: int = 64,
                 W: int = 2048,
                 fov_up: float = 3.0,
                 fov_down: float = -25.0,
                 ignore_index: int = 19) -> None:
        self.H = H
        self.W = W
        self.fov_up = fov_up / 180.0 * np.pi
        self.fov_down = fov_down / 180.0 * np.pi
        self.fov = abs(self.fov_down) + abs(self.fov_up)
        self.ignore_index = ignore_index

    def transform(self, input_dict: dict) -> dict:
        points_numpy = input_dict['points'].numpy()
        input_dict['num_points'] = points_numpy.shape[0]

        proj_image = np.full((self.H, self.W, 4), -1, dtype=np.float32)
        proj_idx = np.full((self.H, self.W), -1, dtype=np.int64)

        # get depth of all points
        depth = np.linalg.norm(points_numpy[:, :3], 2, axis=1)

        # get angles of all points
        yaw = -np.arctan2(points_numpy[:, 1], points_numpy[:, 0])
        pitch = np.arcsin(points_numpy[:, 2] / depth)

        # get projection in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)
        proj_y = 1.0 - (pitch + abs(self.fov_down)) / self.fov

        # scale to image size using angular resolution
        proj_x *= self.W
        proj_y *= self.H

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int64)

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int64)

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        proj_idx[proj_y[order], proj_x[order]] = indices[order]
        proj_image[proj_y[order], proj_x[order]] = points_numpy[order]
        proj_mask = (proj_idx > 0).astype(np.int32)

        if 'pts_semantic_mask' in input_dict:
            pts_semantic_mask = input_dict['pts_semantic_mask']
            proj_sem_label = np.full((self.H, self.W),
                                     self.ignore_index,
                                     dtype=np.int64)
            proj_sem_label[proj_y[order],
                           proj_x[order]] = pts_semantic_mask[order]

        interpolated_points = []
        interpolated_labels = []

        # scan all the pixels
        for y in range(self.H):
            for x in range(self.W):
                # check whether the current pixel is valid
                # if valid, just skip this pixel
                if proj_mask[y, x]:
                    continue

                if (x - 1 >= 0) and (x + 1 < self.W):
                    # only when both of right and left pixels are valid,
                    # the interpolated points will be calculated
                    if proj_mask[y, x - 1] and proj_mask[y, x + 1]:
                        # calculated the potential points
                        mean_points = (proj_image[y, x - 1] +
                                       proj_image[y, x + 1]) / 2
                        # change the current pixel to be valid
                        proj_mask[y, x] = 1
                        proj_image[y, x] = mean_points
                        interpolated_points.append(mean_points)

                        if 'pts_semantic_mask' in input_dict:
                            if proj_sem_label[y,
                                              x - 1] == proj_sem_label[y,
                                                                       x + 1]:
                                # if both pixels share the same semantic label,
                                # then just copy the semantic label
                                cur_label = proj_sem_label[y, x - 1]
                            else:
                                # if they have different labels, we consider it
                                # as boundary and set it as ignored label
                                cur_label = self.ignore_index
                            proj_sem_label[y, x] = cur_label
                            interpolated_labels.append(cur_label)

        # concatenate all the interpolated points and labels
        if len(interpolated_points) > 0:
            interpolated_points = np.array(
                interpolated_points, dtype=np.float32)
            points_numpy = np.concatenate((points_numpy, interpolated_points),
                                          axis=0)
            input_dict['points'] = input_dict['points'].new_point(points_numpy)

            if 'pts_semantic_mask' in input_dict:
                interpolated_labels = np.array(
                    interpolated_labels, dtype=np.int64)
                pts_semantic_mask = np.concatenate(
                    (pts_semantic_mask, interpolated_labels), axis=0)
                input_dict['pts_semantic_mask'] = pts_semantic_mask
        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(H={self.H}, '
        repr_str += f'W={self.W}, '
        repr_str += f'fov_up={self.fov_up}, '
        repr_str += f'fov_down={self.fov_down}, '
        repr_str += f'ignore_index={self.ignore_index})'
        return repr_str


@TRANSFORMS.register_module()
class InstanceCopy(BaseTransform):

    def __init__(self,
                 instance_classes: List[int],
                 pre_transform: Optional[Sequence[dict]] = None,
                 prob: float = 1.0) -> None:
        assert is_list_of(instance_classes, int), \
            'instance_classes should be a list of int'
        self.instance_classes = instance_classes
        self.prob = prob
        if pre_transform is None:
            self.pre_transform = None
        else:
            self.pre_transform = Compose(pre_transform)

    def copy_instance(self, input_dict: dict, mix_results: dict) -> dict:
        points = input_dict['points']
        pts_semantic_mask = input_dict['pts_semantic_mask']

        mix_points = mix_results['points']
        mix_pts_semantic_mask = mix_results['pts_semantic_mask']

        concat_points = [points]
        concat_pts_semantic_mask = [pts_semantic_mask]
        for instance_class in self.instance_classes:
            mix_idx = mix_pts_semantic_mask == instance_class
            concat_points.append(mix_points[mix_idx])
            concat_pts_semantic_mask.append(mix_pts_semantic_mask[mix_idx])
        points = points.cat(concat_points)
        pts_semantic_mask = np.concatenate(concat_pts_semantic_mask, axis=0)

        input_dict['points'] = points
        input_dict['pts_semantic_mask'] = pts_semantic_mask
        return input_dict

    def transform(self, input_dict: dict) -> dict:
        if np.random.rand() > self.prob:
            return input_dict

        assert 'dataset' in input_dict
        dataset = input_dict['dataset']

        # get index of other point cloud
        index = np.random.randint(0, len(dataset))

        mix_results = dataset.get_data_info(index)

        if self.pre_transform is not None:
            # pre_transform may also require dataset
            mix_results.update({'dataset': dataset})
            # before instancecopy need to go through
            # the necessary pre_transform
            mix_results = self.pre_transform(mix_results)
            mix_results.pop('dataset')

        input_dict = self.copy_instance(input_dict, mix_results)
        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(instance_classes={self.instance_classes}, '
        repr_str += f'pre_transform={self.pre_transform}, '
        repr_str += f'prob={self.prob})'
        return repr_str
