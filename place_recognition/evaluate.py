import argparse
import logging
import yaml
from datetime import datetime

import torch.nn as nn
from torch.backends import cudnn

from utils.util import check_makedirs
from datasets.place_recognition_dataset import *
from datasets.dataset_info import *
from utils.train_util import get_num_param


def get_args():
    parser = argparse.ArgumentParser(description='Point Cloud Place Recognition')
    parser.add_argument('--dataset', type=str, default='oxford',
                        help='WHU Data: hankou, campus;' \
                             'Oxford RobotCar: oxford;' \
                             '3-Inhouse: university, residential, business' \
                             'MulRan: sejong, dcc' \
                             'KITTI360: kitti360')
    parser.add_argument('--model_type', type=str, default=None,
                        help='model type: pointnet_vlad, pptnet, pptnet_l2_norm,patch_aug_net , minkloc3d_v2')
    parser.add_argument('--model_config', type=str, default=None, help='for minkloc3d_v2')
    parser.add_argument('--weight', type=str, default=None, help='weight file, *.pth')
    parser.add_argument('--exp_dir', type=str, default='/home/ericxhzou/Code/ppt-net-plus/exp',
                        help='experiment directory')
    args = parser.parse_args()
    cfg = dict()
    if args.model_type == 'pointnet_vlad':
        cfg['config'] = 'configs/pointnet_vlad.yaml'
    elif args.model_type == 'pptnet' or args.model_type == 'pptnet_l2_norm':
        cfg['config'] = 'configs/pptnet_origin.yaml'
    elif args.model_type == 'patch_aug_net':
        cfg['config'] = 'configs/patch_aug_net.yaml'
    elif args.model_type == 'minkloc3d_v2':
        cfg['config'] = 'place_recognition/Minkloc3D_V2/config/config_baseline.txt'
        cfg['model_config'] = 'place_recognition/Minkloc3D_V2/models/minkloc3dv2.txt'
    elif args.model_type == 'egonn':
        cfg['config'] = 'place_recognition/EgoNN/config/config_egonn.txt'
        cfg['model_config'] = 'place_recognition/EgoNN/models/egonn.txt'
    elif args.model_type == 'lcdnet' or args.model_type == 'logg3d_net':
        cfg['config'] = 'None'
    _, file_ext = os.path.splitext(cfg['config'])
    if file_ext == '.yaml':
        cfg = yaml.safe_load(open(cfg['config'], 'r'))
    cfg['dataset'] = args.dataset
    cfg['model_type'] = args.model_type
    if args.model_config is not None:
        cfg['model_config'] = args.model_config
    cfg['weight'] = args.weight
    cfg['exp_dir'] = args.exp_dir
    t_str = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    cfg['log_dir'] = os.path.join(args.exp_dir, args.model_type, 'events', "{}_{}".format(t_str, args.dataset))
    check_makedirs(cfg["log_dir"])
    return cfg


def set_seed(seed=None):
    if seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        print("no seed setting!!!")


def get_logger(log_dir):
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.propagate = False
    check_makedirs(log_dir)
    log_filename = os.path.join(log_dir, "train.log")
    file_handler = logging.FileHandler(filename=log_filename)
    logger.addHandler(file_handler)
    return logger


def get_model(model_type=None):
    global args
    print("model type: ", model_type)
    model = None
    if model_type == "pptnet":
        from place_recognition.pptnet_origin.models import pptnet
        model = pptnet.Network(param=args, use_normalize=False)
    elif model_type == "pptnet_l2_norm":
        from place_recognition.pptnet_origin.models import pptnet
        model = pptnet.Network(param=args, use_normalize=True)
    elif model_type == "pointnet_vlad":
        from place_recognition.pointnet_vlad import PointNetVlad
        model = PointNetVlad.PointNetVlad(global_feat=True, feature_transform=True, max_pool=False,
                                    output_dim=256, num_points=4096)
    elif model_type == "patch_aug_net":
        from place_recognition.patch_aug_net.models import patch_aug_net
        model = patch_aug_net.Network(param=args, use_a2a_recon=True, use_l2_norm=True)
    elif model_type == 'minkloc3d_v2':
        from place_recognition.Minkloc3D_V2.models.model_factory import model_factory
        from place_recognition.Minkloc3D_V2.misc.utils import TrainingParams
        params = TrainingParams(args['config'], args['model_config'])
        model = model_factory(params.model_params)
    elif model_type == 'egonn':
        from place_recognition.EgoNN.models.model_factory import model_factory
        from place_recognition.EgoNN.misc.utils import TrainingParams
        params = TrainingParams(args['config'], args['model_config'])
        model = model_factory(params.model_params)
    elif model_type == 'lcdnet':
        from place_recognition.LCDNet.get_models_cfg import get_model
        from collections import OrderedDict
        saved_params = torch.load(args['weight'], map_location='cpu')
        exp_cfg = saved_params['config']
        exp_cfg.batch_size = 6
        exp_cfg.loop_file = 'loop_GT_4m'
        exp_cfg.head = 'UOTHead'
        model = get_model(exp_cfg, is_training=False)
        renamed_dict = OrderedDict()
        for key in saved_params['state_dict']:
            if not key.startswith('module'):
                renamed_dict = saved_params['state_dict']
                break
            else:
                renamed_dict[key[7:]] = saved_params['state_dict'][key]
        # Convert shape from old OpenPCDet
        if renamed_dict['backbone.backbone.conv_input.0.weight'].shape != model.state_dict()[
            'backbone.backbone.conv_input.0.weight'].shape:
            for key in renamed_dict:
                if key.startswith('backbone.backbone.conv') and key.endswith('weight'):
                    if len(renamed_dict[key].shape) == 5:
                        renamed_dict[key] = renamed_dict[key].permute(-1, 0, 1, 2, 3)
        res = model.load_state_dict(renamed_dict, strict=True)
        if len(res[0]) > 0:
            print(f"WARNING: MISSING {len(res[0])} KEYS, MAYBE WEIGHTS LOADING FAILED")
    elif model_type == 'logg3d_net':
        from place_recognition.LoGG3DNet.models import model_factory
        model = model_factory.model_factory('logg3d1k')
        print('Loading weights: {}'.format(args['weight']))
        checkpoint = torch.load(args['weight'])
        model.load_state_dict(checkpoint['model_state_dict'])
    return model


def load_model(model, weight_file):
    global logger
    logger.info("Load model weight {}".format(weight_file))
    checkpoint = torch.load(weight_file)
    if 'state_dict_encoder' in checkpoint:
        model.load_state_dict(checkpoint['state_dict_encoder'])  # old model use 'state_dict_encoder' !
    else:
        model.load_state_dict(checkpoint)
    return model


def get_device():
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in [0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def run(model, test_dataset, print_query_results=False):
    global args, logger, device
    # update global descriptors, Note: it calls 'model.eval()' within the function
    test_dataset.update_global_descs(model, args, device, 100, stat_time=True, save=True)
    test_dataset.find_and_save_top(model_type=args['model_type'], top_k=300, space_type='feat')
    # get recall@topN
    top_k = 25
    recall = np.zeros(top_k)
    precision = np.zeros(top_k)
    count = 0
    one_percent_recall = []
    tot_lost = []
    recall_dict = test_dataset.get_recall_precision(top_k=top_k)
    self_collected = test_dataset.dataset.data_cfg['self_collected']
    logger.info(">>>>>>>>>>>>>>>>>>>> Evaluation of {} <<<<<<<<<<<<<<<<<<<<".format(args['model_type']))
    for query_trip_idx, ref_trip_idx in recall_dict:
        if ref_trip_idx == query_trip_idx:  # skip itself!
            continue
        pair_recall, pair_precision, pair_opr, lost_num, top_one_per_num, query_results, num_query, num_ref = \
            recall_dict[query_trip_idx, ref_trip_idx]
        if num_query == 0:  # invalid
            continue
        query_trip_name = test_dataset.get_trip_name(query_trip_idx)
        ref_trip_name = test_dataset.get_trip_name(ref_trip_idx)
        if ref_trip_idx == -1:
            if test_dataset.skip_trip_itself:
                ref_trip_name = test_dataset.dataset.trip_names[not query_trip_idx]
            else:
                ref_trip_name = test_dataset.dataset.trip_names
        logger.info("--------------------Recall & Precision Results--------------------")
        logger.info("Recall @topN: query:{}, database:{}, data dir:{}".format(
            query_trip_name, ref_trip_name, test_dataset.dataset.data_dir()))
        logger.info("Num of Query: {}, Num of Ref: {}".format(num_query, num_ref))
        logger.info("Recall@top1~{}: {}".format(top_k, str(pair_recall)))
        logger.info("Recall@top1%(={}): {}".format(top_one_per_num, pair_opr))
        logger.info("Precision@top1~{}: {}".format(top_k, str(pair_precision)))
        if (query_trip_name != "helmet_submap" or ref_trip_name == "helmet_submap") and self_collected:  # only for our own datasets!
            continue
        recall += np.array(pair_recall)
        precision += np.array(pair_precision)
        count += 1
        one_percent_recall.append(pair_opr)
        tot_lost.append(lost_num)
        # query results
        if print_query_results:
            logger.info("--------------------Query Results--------------------")
            logger.info('N: {}'.format(len(query_results)))
            for i in range(len(query_results)):
                logger.info("{}th query: {} state: {} x: {} y: {}".format(i, query_results[i]['query']['file'],
                                                                          query_results[i]['state'],
                                                                          query_results[i]['query']['easting'],
                                                                          query_results[i]['query']['northing']))
                logger.info("true positive: {}".format(query_results[i]['true_pos']['file']))
                for j in range(len(query_results[i]['topN_files'])):
                    logger.info("top{}: {} state: {} x: {} y: {}".format(j, query_results[i]['topN_files'][j]['file'],
                                                                         query_results[i]['topN_states'][j],
                                                                         query_results[i]['topN_files'][j]['easting'],
                                                                         query_results[i]['topN_files'][j]['northing']))
    ave_recall = recall / count
    ave_precision = precision / count
    ave_one_percent_recall = np.mean(one_percent_recall)
    lost_mean = np.mean(tot_lost)
    lost_sum = np.sum(tot_lost)

    # output log info
    logger.info("Average Recall @N: {}".format(str(ave_recall)))
    logger.info("Average Recall @Top 1: {}".format(str(ave_recall[0])))
    logger.info("Average Recall @Top 1%: {}".format(str(ave_one_percent_recall)))
    logger.info("Average Precision @N: {}".format(str(ave_precision)))
    logger.info("lost mean: {}, lost sum: {}".format(lost_mean, lost_sum))
    return ave_one_percent_recall


if __name__ == '__main__':
    global args, logger, device
    # load params
    args = get_args()

    # set logging
    logger = get_logger(args['log_dir'])
    logger.info(args)

    # set seed
    set_seed(seed=123)

    # get device
    device = get_device()

    # load model
    model = get_model(model_type=args['model_type'])
    if args['model_type'] != 'lcdnet' and args['model_type'] != 'logg3d_net':
        model = load_model(model, args['weight'])
    model.to(device)

    # print model
    logger.info("=> creating model {}".format(args['model_type']))
    logger.info(model)
    if torch.cuda.device_count() > 1:
        print("Let's use {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    total = get_num_param(model)
    logger.info("Parameters: {}".format(total))

    # # evaluate on train data
    # t_dataset = PlaceRecognitionDataSet(args['dataset'], for_training=True)
    # if len(t_dataset) > 0:
    #     run(model, t_dataset)
    # evaluate on test data
    t_dataset = PlaceRecognitionDataSet(args['dataset'], for_training=False)
    if len(t_dataset) > 0:
        run(model, t_dataset)