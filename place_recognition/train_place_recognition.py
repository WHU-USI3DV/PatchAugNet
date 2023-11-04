import argparse
import logging
from datetime import datetime

import torch.nn as nn
import yaml
from torch.backends import cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.tensorboard import SummaryWriter

from losses import pointnetvlad_loss
from datasets.place_recognition_dataset import *
from datasets.scene_dataset import *
from datasets.dataset_info import *
from utils.util import check_makedirs
from utils.visualization import *
from utils.train_util import get_num_param, get_flops


def get_args():
    parser = argparse.ArgumentParser(description='Point Cloud Place Recognition')
    parser.add_argument('--config', type=str, default='configs/patch_aug_net.yaml', help='config file')
    parser.add_argument('--dataset', type=str, default='oxford',
                        help='WHU Data: hankou, campus;' \
                             'Oxford RobotCar: oxford;' \
                             '3-Inhouse: university, residential, business' \
                             'MulRan: sejong, dcc' \
                             'KITTI360: kitti360'
                        )
    parser.add_argument('--resume', type=str, default=None, help='resume')
    parser.add_argument('--eval', default=False, action='store_true', help='evaluation of the model')

    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, 'r'))
    cfg["dataset"] =args.dataset
    cfg["resume"] = args.resume
    cfg["eval"] = args.eval
    cfg["eval"] = args.eval
    cfg["event_dir"] = os.path.join(cfg["EXP_DIR"], cfg['model_type'], 'events', "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now()))
    cfg["save_path"] = os.path.join(cfg["EXP_DIR"], cfg['model_type'], 'saved_model')
    check_makedirs(cfg["save_path"])
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
                                    output_dim=args['FEATURE_OUTPUT_DIM'], num_points=args['NUM_POINTS'])
    elif model_type == "patch_aug_net":
        from place_recognition.patch_aug_net.models import patch_aug_net
        model = patch_aug_net.Network(param=args, use_a2a_recon=args['use_patch_recon'], use_l2_norm=True)
    return model


def get_device():
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args["TRAIN_GPU"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def get_current_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def get_loss_func(loss_type=None):
    if loss_type == 'quadruplet':
        return pointnetvlad_loss.quadruplet_loss
    elif loss_type == 'hphn_quadruplet':
        return pointnetvlad_loss.hphn_quadruplet_loss
    elif loss_type == 'contrastive':
        return pointnetvlad_loss.contrastive_loss
    elif loss_type == 'chamfer':
        return pointnetvlad_loss.chamfer_loss
    elif loss_type == 'patch_chamfer':
        return pointnetvlad_loss.patch_chamfer_loss
    elif loss_type == 'emd':
        return pointnetvlad_loss.emd_loss
    elif loss_type == 'patch_emd':
        return pointnetvlad_loss.patch_emd_loss
    elif loss_type == 'point_pair':
        return pointnetvlad_loss.point_pair_loss
    else:
        return pointnetvlad_loss.triplet_loss_wrapper


def get_optimizer(parameters, optimizer_type, learning_rate, momentum=0.9):
    parameters = filter(lambda p: p.requires_grad, parameters)
    if optimizer_type == 'momentum':
        return torch.optim.SGD(parameters, learning_rate, momentum=momentum)
    elif optimizer_type == 'adam':
        return torch.optim.Adam(parameters, learning_rate)
    else:
        return torch.optim.Adam(parameters, learning_rate)


def get_lr_scheduler(lr_decay_type, optimizer, step_size=10, gamma=0.2, max_epoch=10, base_learning_rate=0.1):
    if lr_decay_type == 'step':
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif lr_decay_type == 'cosine':
        return CosineAnnealingLR(optimizer, max_epoch, eta_min=base_learning_rate)
    else:
        return None


def run_model(model, queries, positives, negatives, other_neg, nn_dict=None, num_points=4096, require_grad=True):
    global args, device
    queries_tensor = torch.from_numpy(queries).float()
    positives_tensor = torch.from_numpy(positives).float()
    negatives_tensor = torch.from_numpy(negatives).float()
    other_neg_tensor = torch.from_numpy(other_neg).float()
    feed_tensor = torch.cat((queries_tensor, positives_tensor, negatives_tensor, other_neg_tensor), 1)
    feed_tensor = feed_tensor.view((-1, 1, num_points, 3))  # batch_size x 1 x N x 3
    feed_tensor.requires_grad_(require_grad)
    feed_tensor = feed_tensor.to(device, non_blocking=True)

    global_desc, patch_recon = None, None
    if require_grad:
        if nn_dict is not None:
            global_desc, patch_recon = model.forward(feed_tensor, nn_dict)
        else:
            global_desc = model.forward(feed_tensor)
    else:
        with torch.no_grad():
            if nn_dict is not None:
                global_desc, patch_recon = model.forward(feed_tensor, nn_dict)
            else:
                global_desc = model.forward(feed_tensor)
    batch_size = 1
    global_desc = global_desc.view(queries_tensor.shape[0], -1, args["FEATURE_OUTPUT_DIM"])
    global_desc = torch.split(global_desc, [batch_size, batch_size * args["TRAIN_POSITIVES_PER_QUERY"],
                                            batch_size * args["TRAIN_NEGATIVES_PER_QUERY"], batch_size], dim=1)
    return {'global_desc': global_desc, 'patch_recon': patch_recon}


def save_model(model, epoch, optimizer, i=None, copy_to_event_dir=False):
    global args, logger
    if isinstance(model, nn.DataParallel):
        model_to_save = model.module
    else:
        model_to_save = model

    if i is not None:
        save_name = os.path.join(args["save_path"], "train_epoch_{}_iter{}.pth".format(str(epoch), str(i)))
    else:
        save_name = os.path.join(args["save_path"], "train_epoch_{}_end.pth".format(str(epoch)))
    obj_to_save = {'epoch': epoch, 'iter': TOTAL_ITERATIONS, 'optimizer': optimizer.state_dict(),
                   'state_dict_encoder': model_to_save.state_dict()}  # old model use 'state_dict_encoder' !
    torch.save(obj_to_save, save_name)
    logger.info("Model Saved As {}".format(save_name))
    if copy_to_event_dir:
        import shutil
        shutil.copyfile(save_name, os.path.join(args['event_dir'], "train_epoch_x_end.pth"))


def train_one_epoch(model, optimizer, train_dataset, train_writer, loss_func, epoch):
    global args, logger, device, TOTAL_ITERATIONS
    # params about reconstruction
    use_hard_neg = True
    use_patch_feature_contrast = args['use_patch_feature_contrast']
    use_hard_negative_patch_mining = args['use_hard_negative_patch_mining']
    hard_neg_epoch_for_patch_align = 10
    vis_cloud = False

    # params used during training
    iter_loss = {'place_recognition': [], 'patch_recon_a2a': [], 'patch_recon_a2b': []}
    num_iter_loss = {'place_recognition': 0, 'patch_recon_a2a': 0, 'patch_recon_a2b': 0}
    loss_alpha = {'place_recognition': args['weight_place_recognition'],
                  'patch_recon_a2a': args['weight_patch_recon'],
                  'patch_recon_a2b': args['weight_patch_feature_contrast']}
    logger.info("loss_alpha: {}".format(loss_alpha))

    # clear global descriptors
    hard_neg_epoch = 5
    if use_hard_neg:
        if epoch <= hard_neg_epoch:
            train_dataset.clear_global_descs()
    else:
        train_dataset.clear_global_descs()

    # train one batch
    batch_size = args["TRAIN_BATCH_SIZE"]
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=mycollate)
    count = 0
    for data_item in tqdm(train_loader):
        torch.cuda.empty_cache()
        faulty_tuple = False
        no_other_neg = False
        queries, positives, negatives, other_neg = [], [], [], []
        nn_dict_list = []
        for tuple_j in data_item:
            query_idx_in_dataset, positive_indices, _, _ = tuple_j['indices']
            query_idx_in_dataset = query_idx_in_dataset[0]
            query_pcs_j, pos_pcs_j, neg_pcs_j, neg2_pcs_j = tuple_j['input_cloud']
            if len(pos_pcs_j) < args["TRAIN_POSITIVES_PER_QUERY"] or len(neg_pcs_j) < args["TRAIN_NEGATIVES_PER_QUERY"]:
                faulty_tuple = True
                break
            if len(neg2_pcs_j) == 0:  # For Quadruplet Loss
                no_other_neg = True
                break
            queries.append(query_pcs_j)
            positives.append(pos_pcs_j)
            negatives.append(neg_pcs_j)
            other_neg.append(neg2_pcs_j)
            # load overlap indices
            tuple_j['overlap_indices'] = train_dataset.dataset.get_overlap_indices(query_idx_in_dataset, positive_indices)
            nn_dict_list.append(tuple_j['overlap_indices'])
        # skip if this tuple is faulty (no enough positives or negatives)
        if faulty_tuple:
            continue
        # no other negative for Quadruplet Loss
        if no_other_neg:
            continue
        queries = np.array(queries, dtype=np.float32)
        positives = np.array(positives, dtype=np.float32)
        negatives = np.array(negatives, dtype=np.float32)
        other_neg = np.array(other_neg, dtype=np.float32)
        # skip if empty
        if len(queries.shape) != 4:
            continue

        # stack elements in nn_dict_list
        nn_dict = dict()
        tuple_size = 1 + args["TRAIN_POSITIVES_PER_QUERY"] + args["TRAIN_NEGATIVES_PER_QUERY"] + 1
        for j in range(len(nn_dict_list)):
            if nn_dict_list[j] is None:
                continue
            for q_idx, p_idx in nn_dict_list[j]:
                nn_dict[q_idx + j * tuple_size, p_idx + j * tuple_size] = nn_dict_list[j][q_idx, p_idx]

        model.train()
        optimizer.zero_grad(set_to_none=True)
        # torch.cuda.synchronize()  # for time cost statistic
        if not args['use_patch_recon'] and not args['use_patch_feature_contrast']:
            nn_dict = None
        if args["model_type"] != 'patch_aug_net':
            nn_dict = None
        out = run_model(model, queries, positives, negatives, other_neg, nn_dict)
        out_queries, out_positives, out_negatives, out_other_neg = out['global_desc']
        patches_recon_data = out['patch_recon']
        # torch.cuda.synchronize()
        cur_loss = dict()
        cur_loss['place_recognition'] = loss_func['place_recognition'](out_queries, out_positives, out_negatives,
                                                                       out_other_neg,
                                                                       args["MARGIN_1"], args["MARGIN_2"],
                                                                       use_min=args["TRIPLET_USE_BEST_POSITIVES"],
                                                                       lazy=args["LOSS_LAZY"],
                                                                       ignore_zero_loss=args["LOSS_IGNORE_ZERO_BATCH"])
        num_iter_loss['place_recognition'] += 1
        # patch augmentation
        if patches_recon_data is not None:
            cloud_indices = patches_recon_data['cloud_indices']
            center_indices = patches_recon_data['center_indices']
            patch_features = patches_recon_data['patch_features']
            origin_patches = patches_recon_data['origin_patches']
            reconstructed_patches = patches_recon_data['reconstructed_patches']
            # vis cloud
            if vis_cloud:
                out_dir = '/home/ericxhzou/Data/recon_clouds'
                origin_q_pc = torch.index_select(origin_patches[0].reshape([-1, 3]), dim=0, index=torch.tensor(np.arange(0, 20480, 5)).to(device))
                origin_q_pc = origin_q_pc.cpu().detach().numpy()
                recon_q_pc = torch.index_select(reconstructed_patches[0].reshape([-1, 3]), dim=0, index=torch.tensor(np.arange(0, 20480, 5)).to(device))
                recon_q_pc = recon_q_pc.cpu().detach().numpy()
                vis_cloud_simple("recon_cloud", [origin_q_pc, recon_q_pc])
                np.save(os.path.join(out_dir, 'origin.bin'), origin_q_pc)
                np.save(os.path.join(out_dir, 'recon.bin'), recon_q_pc)
            # patch reconstruction a2a loss
            if model.use_a2a_recon:
                cur_loss['patch_recon_a2a'] = loss_func['patch_recon_a2a'](origin_patches, reconstructed_patches)
                num_iter_loss['patch_recon_a2a'] += 1
            # patch reconstruction a2b loss
            if use_patch_feature_contrast:
                cur_loss['patch_recon_a2b'] = 0.0
                count_cur_loss = 0
                for m, n in nn_dict:
                    # patches
                    m_center_indices, m_patch_features, m_origin_patches, m_recon_patches = None, None, None, None
                    for k in range(len(cloud_indices)):
                        if cloud_indices[k] == m:
                            m_center_indices = center_indices[k].squeeze().cpu().detach().numpy()
                            m_patch_features = patch_features[k]
                            break
                    n_center_indices, n_patch_features, n_origin_patches, n_recon_patches = None, None, None, None
                    for k in range(len(cloud_indices)):
                        if cloud_indices[k] == n:
                            n_center_indices = center_indices[k].squeeze().cpu().detach().numpy()
                            n_patch_features = patch_features[k]
                            break
                    # indices
                    indices1, pos_indices2, neg_indices2 = [], [], []
                    overlap_indices = nn_dict[m, n]
                    overlap_indices_list = []
                    for k in overlap_indices:
                        overlap_indices_list.append(k)
                    k_list = [n for n in range(len(overlap_indices_list))]
                    if len(k_list) > 500:
                        k_list = random.sample(k_list, 500)
                    for k in k_list:
                        idx1 = np.where(m_center_indices == overlap_indices_list[k].idx1)[0].tolist()
                        if len(idx1) == 0:
                            continue
                        list_near_indices = []
                        for near_idx in overlap_indices_list[k].near_indices2:
                            list_near_indices.append(near_idx)
                        pos_idx2 = np.where(np.isin(n_center_indices, list_near_indices))[0].tolist()
                        if len(pos_idx2) == 0:
                            continue
                        list_far_indices = []
                        if epoch > hard_neg_epoch_for_patch_align and use_hard_negative_patch_mining:  # use hard negative patches only
                            for far_idx in overlap_indices_list[k].bad_far_indices2:
                                list_far_indices.append(far_idx)
                        else:
                            temp_list_far_indices = []
                            for far_idx in overlap_indices_list[k].far_indices2:
                                temp_list_far_indices.append(far_idx)
                            for far_idx in overlap_indices_list[k].bad_far_indices2:
                                temp_list_far_indices.append(far_idx)
                            for far_i in range(0, len(temp_list_far_indices), 2):
                                list_far_indices = temp_list_far_indices[far_i]
                        neg_idx2 = np.where(np.isin(n_center_indices, list_far_indices))[0].tolist()
                        if len(neg_idx2) == 0:
                            continue
                        idx1 = (np.ones(len(pos_idx2), dtype='int32') * idx1[0]).tolist()
                        neg_idx2 = np.random.choice(neg_idx2, len(pos_idx2), replace=True).tolist()
                        indices1 += idx1
                        pos_indices2 += pos_idx2
                        neg_indices2 += neg_idx2
                    # related patches and its features
                    q_patch_features_selected, q_origin_patches_selected, q_recon_patches_selected = [], [], []
                    p_patch_features_selected, p_origin_patches_selected, p_recon_patches_selected = [], [], []
                    n_patch_features_selected, n_origin_patches_selected, n_recon_patches_selected = [], [], []
                    for k in range(len(indices1)):
                        # query
                        idx1 = torch.tensor([indices1[k]]).to(device, non_blocking=True)
                        q_patch_features_selected.append(torch.index_select(m_patch_features, dim=0, index=idx1).squeeze())
                        # positive
                        pos_idx2 = torch.tensor([pos_indices2[k]]).to(device, non_blocking=True)
                        p_patch_features_selected.append(torch.index_select(n_patch_features, dim=0, index=pos_idx2).squeeze())
                        # negative
                        neg_idx2 = torch.tensor([neg_indices2[k]]).to(device, non_blocking=True)
                        n_patch_features_selected.append(torch.index_select(n_patch_features, dim=0, index=neg_idx2).squeeze())
                    if len(q_patch_features_selected) == 0:
                        continue
                    temp_loss = loss_func['patch_recon_a2b'](q_patch_features_selected, p_patch_features_selected, n_patch_features_selected, args["MARGIN_1"])
                    cur_loss['patch_recon_a2b'] += temp_loss
                    count_cur_loss += 1
                if count_cur_loss > 0:
                    cur_loss['patch_recon_a2b'] /= count_cur_loss
                    num_iter_loss['patch_recon_a2b'] += 1
        loss_sum = 0.0
        for key in cur_loss:
            cur_loss[key] = cur_loss[key] * loss_alpha[key]
            loss_sum += cur_loss[key]
        if loss_sum > 1e-10:
            loss_sum.backward()
            optimizer.step()

        TOTAL_ITERATIONS += batch_size

        # calculate remain time
        for key in cur_loss:
            iter_loss[key].append(float(cur_loss[key]))
            train_writer.add_scalars("iter_loss", {"train_{}".format(key): float(cur_loss[key])}, TOTAL_ITERATIONS)

        # use hard negative mining when the model is robust enough
        count += 1
        if epoch > hard_neg_epoch and use_hard_neg:
            if count % (1400 // batch_size) == 29:
                train_dataset.update_global_descs(model, args, device, batch_size=36)
                logger.info("Updated cached feature vectors and Prepare to use hard negative mining!")

    epoch_loss = dict()
    for key in iter_loss:
        if num_iter_loss[key] == 0:
            epoch_loss[key] = 0.0
            continue
        epoch_loss[key] = np.sum(iter_loss[key]) / num_iter_loss[key]
        train_writer.add_scalars("epoch_loss", {"train_{}".format(key): epoch_loss[key]}, epoch + 1)
    save_model(model, epoch, optimizer, copy_to_event_dir=True)
    return epoch_loss


def eval(model, test_dataset, eval_writer, epoch, eval_name, print_query_results=False):
    r""" for evaluate test at each epoch """
    global args, logger, device
    # update global descriptors
    test_dataset.update_global_descs(model, args, device, batch_size=36, stat_time=True)
    # get recall@topN
    top_k = 25
    recall = np.zeros(top_k)
    precision = np.zeros(top_k)
    count = 0
    one_percent_recall = []
    tot_lost = []
    recall_dict = test_dataset.get_recall_precision(top_k=top_k)
    self_collected = test_dataset.dataset.data_cfg['self_collected']
    logger.info(">>>>>>>>>>>>>>>>>>>> Evaluation of {} <<<<<<<<<<<<<<<<<<<<".format(eval_name))
    for query_trip_idx, ref_trip_idx in recall_dict:
        if ref_trip_idx == query_trip_idx:  # skip itself!
            continue
        pair_recall, pair_precision, pair_opr, lost_num, top_one_per_num, query_results, num_query, num_ref = recall_dict[
            query_trip_idx, ref_trip_idx]
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
            query_trip_name, ref_trip_name, test_dataset.dataset.data_dir))
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
                logger.info("{}th query: {} state: {} x: {} y: {}".format(i, query_results[i]['query']['file'], query_results[i]['state'],
                                                              query_results[i]['query']['easting'], query_results[i]['query']['northing']))
                logger.info("true positive: {}".format(query_results[i]['true_pos']['file']))
                for j in range(len(query_results[i]['topN_files'])):
                    logger.info("top{}: {} state: {} x: {} y: {}".format(j, query_results[i]['topN_files'][j]['file'], query_results[i]['topN_states'][j],
                                                             query_results[i]['topN_files'][j]['easting'], query_results[i]['topN_files'][j]['northing']))
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
    eval_writer.add_scalars("ave_recall@topN", {"top1_{}".format(eval_name): ave_recall[0]}, epoch + 1)
    eval_writer.add_scalars("ave_recall@topN", {"top1%_{}".format(eval_name): ave_one_percent_recall}, epoch + 1)
    eval_writer.add_scalars("ave_precision@topN", {"top{}_{}".format(top_k, eval_name): ave_precision[-1]}, epoch + 1)
    return ave_one_percent_recall


def main_work():
    global args, logger, device, TOTAL_ITERATIONS

    # set logging
    logger = get_logger(args['event_dir'])
    logger.info(args)

    # load datasets
    train_dataset = PlaceRecognitionDataSet(args['dataset'], for_training=True,
                                            num_pos=args['TRAIN_POSITIVES_PER_QUERY'],
                                            num_neg=args['TRAIN_NEGATIVES_PER_QUERY'],
                                            skip_trip_itself=False, load_overlap_indices=False)
    test_dataset = PlaceRecognitionDataSet(args['dataset'], for_training=False,
                                           num_pos=args['TRAIN_POSITIVES_PER_QUERY'],
                                           num_neg=args['TRAIN_NEGATIVES_PER_QUERY'],
                                           skip_trip_itself=True)

    # get model
    model = get_model(model_type=args["model_type"])
    model.to(device)

    # setup optimizer
    base_learning_rate = args["BASE_LEARNING_RATE"]
    parameters = model.parameters()
    optimizer = get_optimizer(parameters, args["OPTIMIZER"], base_learning_rate, args["MOMENTUM"])

    # resume from existing model
    if args["resume"] is not None:
        resume_filename = os.path.join(args["save_path"], args["resume"])
        if os.path.exists(args["resume"]):
            resume_filename = args["resume"]
        logger.info("Resuming From {}".format(resume_filename))
        checkpoint = torch.load(resume_filename)
        starting_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict_encoder'])  # old model use 'state_dict_encoder' !
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        starting_epoch = 0

    # learning rate schedule
    lr_scheduler = get_lr_scheduler(args["LEARNING_RATE_DECAY"],
                                    optimizer,
                                    step_size=10,
                                    gamma=0.2,
                                    max_epoch=args["MAX_EPOCH"],
                                    base_learning_rate=base_learning_rate)

    # set loss function
    loss_func = {'place_recognition': get_loss_func(args["LOSS_FUNCTION"]),
                 'patch_recon_a2a': get_loss_func("patch_chamfer"),
                 'patch_recon_a2b': get_loss_func("contrastive")}

    # print model
    logger.info("=> creating model ...")
    logger.info(model)
    if torch.cuda.device_count() > 1:
        print("Let's use {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    total = get_num_param(model)
    logger.info("Parameters: {}".format(total))
    feed_tensor = torch.rand(4, 1, 4096, 3).to(device)
    flops = get_flops(model, feed_tensor)
    logger.info("FLOPs: {:.4f}GFLOPs".format(flops / 4))

    # set summary writer for tensorboard
    event_writer = SummaryWriter(args['event_dir'])

    # start training
    TOTAL_ITERATIONS = 0
    if args["eval"] is not True:
        for epoch in range(starting_epoch, args["MAX_EPOCH"]):
            logger.info('**** EPOCH {:03d} ****'.format(epoch))
            train_one_epoch(model, optimizer, train_dataset, event_writer, loss_func, epoch)
            # eval(model, train_dataset, event_writer, epoch, 'train')
            # if epoch % 5 == 0 or epoch == args["MAX_EPOCH"]:
            #     eval(model, test_dataset, event_writer, epoch, 'testing')
            if lr_scheduler is not None:
                lr_scheduler.step()
    else:
        eval(model, test_dataset, event_writer, starting_epoch, 'testing', print_query_results=False)
    event_writer.close()


if __name__ == '__main__':
    # load params
    args = get_args()

    # set seed
    set_seed(seed=args["MANUAL_SEED"])

    # get device
    device = get_device()

    main_work()
