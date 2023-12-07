import argparse
import os
from os import path as osp
from pathlib import Path

import mmengine
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits


def create_nuscenes_infos(root_path: str, info_prefix: str,
                          version: str) -> None:
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown')

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    test = 'test' in version
    if test:
        print(f'test scene: {len(train_scenes)}')
    else:
        print(f'train scene: {len(train_scenes)}, '
              f'val scene: {len(val_scenes)}')
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, train_scenes, val_scenes, test)

    metainfo = dict(dataset='nuscenes', version=version)
    if test:
        print(f'test sample: {len(train_nusc_infos)}')
        test_data = dict(metainfo=metainfo, data_list=train_nusc_infos)
        info_test_path = osp.join(root_path, f'{info_prefix}_infos_test.pkl')
        mmengine.dump(test_data, info_test_path)
    else:
        print(f'train sample: {len(train_nusc_infos)}, '
              f'val sample: {len(val_nusc_infos)}')
        train_data = dict(metainfo=metainfo, data_list=train_nusc_infos)
        info_train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
        mmengine.dump(train_data, info_train_path)
        val_data = dict(metainfo=metainfo, data_list=val_nusc_infos)
        info_val_path = osp.join(root_path, f'{info_prefix}_infos_val.pkl')
        mmengine.dump(val_data, info_val_path)


def get_available_scenes(nusc: NuScenes) -> list:
    available_scenes = []
    print(f'total scene num: {len(nusc.scene)}')
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, _, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
            if not mmengine.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print(f'exist scene num: {len(available_scenes)}')
    return available_scenes


def _fill_trainval_infos(nusc: NuScenes,
                         train_scenes: list,
                         val_scenes: list,
                         test: bool = False) -> tuple:
    train_nusc_infos = []
    val_nusc_infos = []

    for sample in mmengine.track_iter_progress(nusc.sample):
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_path, _, _ = nusc.get_sample_data(lidar_token)
        mmengine.check_file_exist(lidar_path)

        info = {
            'lidar_points': {
                'lidar_path': Path(lidar_path).name,
                'num_pts_feats': 5,
                'sample_data_token': lidar_token
            },
            'token': sample['token'],
        }

        if not test:
            pts_semantic_mask_path = osp.join(
                nusc.dataroot,
                nusc.get('lidarseg', lidar_token)['filename'])
            info['pts_semantic_mask_path'] = Path(pts_semantic_mask_path).name

        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        '--root-path',
        type=str,
        default='./data/nuscenes',
        help='specify the root path of dataset')
    parser.add_argument(
        '--version',
        type=str,
        default='v1.0',
        required=False,
        help='specify the dataset version')
    parser.add_argument('--extra-tag', type=str, default='nuscenes')
    args = parser.parse_args()

    if args.version != 'v1.0-mini':
        create_nuscenes_infos(args.root_path, args.extra_tag,
                              f'{args.version}-trainval')
        create_nuscenes_infos(args.root_path, args.extra_tag,
                              f'{args.version}-test')
    else:
        create_nuscenes_infos(args.root_path, args.extra_tag, args.version)
