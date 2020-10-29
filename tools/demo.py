import argparse
import glob
from pathlib import Path

import mayavi.mlab as mlab
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils import visualize_utils as V


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

#创建一个ArgumentParser对象，格式: 参数名,类型,默认值, 帮助信息
def parse_config(): 
    parser = argparse.ArgumentParser(description='arg parser')
    #设置模型参数
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    #设置数据路径参数
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    #设置预训练模型，Check point 校验点 在系统运行中当出现查找数据请求时，
    #系统从数据库中找出这些数据并存入内存区，这样用户就可以对这些内存区数据进行修改等
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    # extension 指定点云数据文件的扩展名
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    #arguments参数，configuration配制
    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger() #logger记录日志
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    #建立一个DemoDataset类，其中储存关于输入数据的所有信息，包含六个参数
    # dataset_cfg=cfg.DATA_CONFIG # 数据参数
    # 包含数据集 / 数据路径 / 信息路径 / 数据处理器 / 数据增强器等
    # class_names=cfg.CLASS_NAMES # 类别名
    # training=False # 是否训练
    # root_path=Path(args.data_path) # 数据路径
    # ext=args.ext # 扩展
    # logger=logger # 日志
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    ) 

    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad(): #目的是使得其中的数据不需要计算梯度，也不会进行反向传播
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )
            mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
