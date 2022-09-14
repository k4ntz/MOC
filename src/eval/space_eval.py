from utils import MetricLogger
import numpy as np
import torch
import os
import sys
from datetime import datetime
from torch.utils.data import Subset, DataLoader
import os.path as osp
from tqdm import tqdm
from .eval_cfg import eval_cfg
from .ap import read_boxes, convert_to_boxes, compute_ap, compute_counts, compute_prec_rec
from .kalman_filter import classify_encodings
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from .classify_z_what import evaluate_z_what
import PIL
from torchvision.utils import draw_bounding_boxes as draw_bb
import os
import pprint
import pickle
from .utils import flatten


class SpaceEval:
    def __init__(self):
        self.relevant_object_hover_path = None
        self.eval_file = None
        self.eval_file_path = None
        self.first_eval = True

    # @torch.no_grad()
    # def test_eval(self, model, testset, bb_path, device, evaldir, info, global_step, cfg):
    #     losses, logs = self.apply_model(testset, device, model, global_step)
    #     result_dict = self.eval_ap_and_acc(logs, testset, bb_path)
    #     clustering_result_dict = self.eval_clustering(logs, testset, global_step, cfg)
    #     os.makedirs(evaldir, exist_ok=True)
    #     path = osp.join(evaldir, 'results_{}.json'.format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    #     self.save_to_json(result_dict, path, info)
    #     self.print_result(result_dict, [sys.stdout, open('./results.txt', 'w')])

    def write_metric(self, writer, tb_label, value, global_step, use_writer=True, make_sep=True):
        if use_writer:
            writer.add_scalar(tb_label, value, global_step)
        self.eval_file.write(f'{value};' if make_sep else f'{value}')

    def write_header(self):
        columns = ['global_step']
        if 'cluster' in eval_cfg.train.metrics:
            for class_name in ['all', 'moving', 'relevant']:
                for training_objects_per_class in [1, 4, 16, 64]:
                    columns.append(f'{class_name}_few_shot_accuracy_with_{training_objects_per_class}')
                columns.append(f'{class_name}_few_shot_accuracy_cluster_nn')
                columns.append(f'{class_name}_adjusted_mutual_info_score')
                columns.append(f'{class_name}_adjusted_rand_score')
            columns.append(f'relevant_bayes_accuracy')
        if 'mse' in eval_cfg.train.metrics:
            columns.append('mse')
        if 'ap' in eval_cfg.train.metrics:
            for class_name in ['all', 'moving', 'relevant']:
                for iou_t in np.linspace(0.1, 0.9, 9):
                    columns.append(f'{class_name}_ap_{iou_t:.2f}')
                columns.append(f'{class_name}_accuracy')
                columns.append(f'{class_name}_perfect')
                columns.append(f'{class_name}_overcount')
                columns.append(f'{class_name}_undercount')
                columns.append(f'{class_name}_error_rate')
                columns.append(f'{class_name}_precision')
                columns.append(f'{class_name}_recall')

        with open(self.eval_file_path, "w") as file:
            file.write(";".join(columns))
            file.write("\n")

    @torch.no_grad()
    # @profile
    def train_eval(self, model, valset, bb_path, writer, global_step, device, checkpoint, checkpointer, cfg):
        """
        Evaluation during training. This includes:
            - mse evaluated on validation set
            - ap and accuracy evaluated on validation set
            - cluster metrics evaluated on validation set
        :return:
        """
        if self.first_eval:
            self.first_eval = False
            self.eval_file_path = f'{cfg.logdir}/{cfg.exp_name}/metrics.csv'
            self.relevant_object_hover_path = f'{cfg.logdir}/{cfg.exp_name}/hover'
            if os.path.exists(self.eval_file_path):
                os.remove(self.eval_file_path)
            os.makedirs(self.relevant_object_hover_path, exist_ok=True)
            os.makedirs(self.relevant_object_hover_path + "Img", exist_ok=True)
            self.write_header()

        losses, logs = self.apply_model(valset, device, model, global_step)
        with open(self.eval_file_path, "a") as self.eval_file:
            self.write_metric(None, None, global_step, global_step, use_writer=False)
            if 'cluster' in eval_cfg.train.metrics:
                results = self.train_eval_clustering(logs, valset, writer, global_step, cfg)
                if cfg.train.log:
                    pp = pprint.PrettyPrinter(depth=2)
                    for res in results:
                        print("Cluster Result:")
                        pp.pprint(results[res])
                checkpointer.save_best('rand_score_relevant',
                                       results['relevant'][0]['adjusted_rand_score'],
                                       checkpoint, min_is_better=False)
            if 'mse' in eval_cfg.train.metrics:
                mse = self.train_eval_mse(logs, losses, writer, global_step)
                print("MSE result: ", mse)
            if 'ap' in eval_cfg.train.metrics:
                results = self.train_eval_ap_and_acc(logs, valset, bb_path, writer, global_step)
                checkpointer.save_best('accuracy', results['accuracy_relevant'], checkpoint, min_is_better=False)
                if cfg.train.log:
                    results = {k2: v2[len(v2) // 4] if isinstance(v2, list) or isinstance(v2, np.ndarray) else v2 for
                               k2, v2, in
                               results.items()}
                    pp = pprint.PrettyPrinter(depth=2)
                    print("AP Result:")
                    pp.pprint({k: v for k, v in results.items() if "iou" not in k})
            self.eval_file.write("\n")

    @torch.no_grad()
    def test_eval(self, model, testset, bb_path, writer, global_step, device, checkpoint, checkpointer, cfg):
        """
        Evaluation during training. This includes:
            - mse evaluated on validation set
            - ap and accuracy evaluated on validation set
            - cluster metrics evaluated on validation set
        :return:
        """
        # make checkpoint test dir
        chpt_dir_save = checkpointer.checkpointdir
        checkpointer.checkpointdir = chpt_dir_save.replace("/eval/", "/test_eval/") + f"_{model.module.arch_type}"
        os.makedirs(checkpointer.checkpointdir, exist_ok=True)
        efp_save = self.eval_file_path
        rohp_save = self.relevant_object_hover_path
        self.eval_file_path = f'../final_test_results/{cfg.exp_name}_{model.module.arch_type}_seed{cfg.seed}_metrics.csv'
        self.relevant_object_hover_path = f'../final_test_results/{cfg.exp_name}/hover'
        if os.path.exists(self.eval_file_path):
            os.remove(self.eval_file_path)
        os.makedirs(self.relevant_object_hover_path, exist_ok=True)
        os.makedirs(f'../final_test_results/{cfg.exp_name}/hover', exist_ok=True)
        os.makedirs(os.path.join(self.relevant_object_hover_path, "img"), exist_ok=True)
        self.write_header()
        losses, logs = self.apply_model(testset, device, model, global_step)
        with open(self.eval_file_path, "a") as self.eval_file:
            self.write_metric(None, None, global_step, global_step, use_writer=False)
            if 'cluster' in eval_cfg.train.metrics:
                results = self.train_eval_clustering(logs, testset, writer, global_step, cfg)
                if cfg.train.log:
                    pp = pprint.PrettyPrinter(depth=2)
                    for res in results:
                        print("Cluster Result:")
                        pp.pprint(results[res])
                # checkpointer.save_best('rand_score_relevant',
                #                        results['relevant'][0]['adjusted_rand_score'],
                #                        checkpoint, min_is_better=False)
            if 'mse' in eval_cfg.train.metrics:
                mse = self.train_eval_mse(logs, losses, writer, global_step)
                print("MSE result: ", mse)
            if 'ap' in eval_cfg.train.metrics:
                results = self.train_eval_ap_and_acc(logs, testset, bb_path, writer, global_step)
                # checkpointer.save_best('accuracy', results['accuracy_relevant'], checkpoint, min_is_better=False)
                if cfg.train.log:
                    results = {k2: v2[len(v2) // 4] if isinstance(v2, list) or isinstance(v2, np.ndarray) else v2 for
                               k2, v2, in
                               results.items()}
                    pp = pprint.PrettyPrinter(depth=2)
                    print("AP Result:")
                    pp.pprint({k: v for k, v in results.items() if "iou" not in k})
            self.eval_file.write("\n")
        print(f"Saved results in {self.eval_file_path}")
        self.eval_file_path = efp_save
        self.relevant_object_hover_path = rohp_save
        checkpointer.checkpointdir = chpt_dir_save

    @torch.no_grad()
    def apply_model(self, dataset, device, model, global_step, use_global_step=False):
        print('Applying the model for evaluation...')
        model.eval()
        if eval_cfg.train.num_samples:
            num_samples = max(eval_cfg.train.num_samples.values())
        else:
            num_samples = len(dataset)
        batch_size = eval_cfg.train.batch_size
        num_workers = eval_cfg.train.num_workers
        data_subset = Subset(dataset, indices=range(num_samples))
        dataloader = DataLoader(data_subset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        losses = []
        logs = []
        with torch.no_grad():
            for imgs, motion, motion_z_pres, motion_z_where in dataloader:
                imgs = imgs.to(device)
                motion = None
                motion_z_pres = None
                motion_z_where = None
                # motion = motion.to(device)
                # motion_z_pres = motion_z_pres.to(device)
                # motion_z_where = motion_z_where.to(device)
                loss, log = model(imgs, motion, motion_z_pres, motion_z_where,
                                  global_step if use_global_step else 1000000000)
                for key in ['imgs', 'y', 'log_like', 'loss', 'fg', 'z_pres_prob_pure',
                            'prior_z_pres_prob', 'o_att', 'alpha_att_hat', 'alpha_att', 'alpha_map', 'alpha_map_pure',
                            'importance_map_full_res_norm', 'kl_z_what', 'kl_z_pres', 'kl_z_scale', 'kl_z_shift',
                            'kl_z_depth', 'kl_z_where', 'comps', 'masks', 'bg', 'kl_bg']:
                    del log[key]
                losses.append(loss)
                logs.append(log)
        model.train()
        return losses, logs

    @torch.no_grad()
    def train_eval_ap_and_acc(self, logs, valset, bb_path, writer: SummaryWriter, global_step):
        """
        Evaluate ap and accuracy during training
        :return: result_dict
        """
        result_dict = self.eval_ap_and_acc(logs, valset, bb_path)
        for class_name in ['all', 'moving', 'relevant']:
            APs = result_dict[f'APs_{class_name}']
            iou_thresholds = result_dict[f'iou_thresholds_{class_name}']
            accuracy = result_dict[f'accuracy_{class_name}']
            perfect = result_dict[f'perfect_{class_name}']
            overcount = result_dict[f'overcount_{class_name}']
            undercount = result_dict[f'undercount_{class_name}']
            error_rate = result_dict[f'error_rate_{class_name}']
            precision = result_dict[f'precision_{class_name}']
            recall = result_dict[f'recall_{class_name}']

            for ap in APs:
                self.write_metric(None, 'ignored', ap, global_step, use_writer=False)
            for ap, thres in zip(APs[1::4], iou_thresholds[1::4]):
                writer.add_scalar(f'val_aps_{class_name}/ap_{thres:.1}', ap, global_step)
            writer.add_scalar(f'{class_name}/ap_avg_0.5', APs[len(APs) // 2], global_step)
            writer.add_scalar(f'{class_name}/ap_avg_up', np.mean(APs[len(APs) // 2:]), global_step)
            writer.add_scalar(f'{class_name}/ap_avg', np.mean(APs), global_step)
            self.write_metric(writer, f'{class_name}/accuracy', accuracy, global_step)
            self.write_metric(writer, f'{class_name}/perfect', perfect, global_step)
            self.write_metric(writer, f'{class_name}/overcount', overcount, global_step)
            self.write_metric(writer, f'{class_name}/undercount', undercount, global_step)
            self.write_metric(writer, f'{class_name}/error_rate', error_rate, global_step)
            self.write_metric(writer, f'{class_name}/precision', precision, global_step)
            self.write_metric(writer, f'{class_name}/recall', recall, global_step,
                              make_sep=class_name != 'relevant')
        return result_dict

    @torch.no_grad()
    def train_eval_mse(self, logs, losses, writer, global_step):
        """
        Evaluate MSE during training
        """
        print('Computing MSE...')
        num_batches = eval_cfg.train.num_samples.mse // eval_cfg.train.batch_size
        metric_logger = MetricLogger()
        for loss, log in zip(losses[:num_batches], logs):
            metric_logger.update(mse=torch.mean(log['mse']))
        mse = metric_logger['mse'].global_avg
        self.write_metric(writer, f'all/mse', mse, global_step=global_step)
        return mse

    @torch.no_grad()
    def train_eval_clustering(self, logs, valset, writer: SummaryWriter, global_step, cfg):
        """
        Evaluate clustering during training
        :return: result_dict
        """
        print('Computing clustering and few-shot linear classifiers...')
        results = self.eval_clustering(logs, valset, global_step, cfg)

        for name, (result_dict, img_path, few_shot_accuracy) in results.items():
            try:
                writer.add_image(f'Clustering PCA {name.title()}', np.array(Image.open(img_path)), global_step,
                                 dataformats='HWC')
            except:
                pass
            for train_objects_per_class in [1, 4, 16, 64]:
                self.write_metric(writer, f'{name}/few_shot_accuracy_with_{train_objects_per_class}',
                                  few_shot_accuracy[f'few_shot_accuracy_with_{train_objects_per_class}'], global_step)
            self.write_metric(writer, f'{name}/few_shot_accuracy_cluster_nn',
                              few_shot_accuracy[f'few_shot_accuracy_cluster_nn'], global_step)
            self.write_metric(writer, f'{name}/adjusted_mutual_info_score',
                              result_dict[f'adjusted_mutual_info_score'], global_step)
            self.write_metric(writer, f'{name}/adjusted_rand_score',
                              result_dict[f'adjusted_rand_score'], global_step)
            if "relevant" in name:
                self.write_metric(writer, f'{name}/bayes_accuracy',
                                  few_shot_accuracy[f'bayes_accuracy'], global_step)
        return results

    # @profile
    @torch.no_grad()
    def eval_clustering(self, logs, dataset, global_step, cfg):
        """
        Evaluate clustering metrics

        :param logs: results from applying the model
        :param dataset: dataset
        :param cfg: config
        :param global_step: gradient step number
        :return metrics: for all classes of evaluation many metrics describing the clustering,
            based on different ground truths
        """
        z_encs = []
        z_whats = []
        all_labels = []
        all_labels_moving = []
        image_refs = []
        batch_size = eval_cfg.train.batch_size
        img_path = os.path.join(dataset.image_path, dataset.game, dataset.mode)
        for i, img in enumerate(logs):
            z_where, z_pres_prob, z_what = img['z_where'], img['z_pres_prob'], img['z_what']
            z_pres_prob = z_pres_prob.squeeze()
            z_pres = z_pres_prob > 0.5
            print(f'{z_pres.sum() / batch_size}')
            if not (0.05 <= z_pres.sum() / batch_size <= 50 * 4):
                z_whats = None
                break
            if cfg.save_relevant_objects:
                for idx, (sel, bbs) in enumerate(zip(z_pres, z_where)):
                    for obj_idx, bb in enumerate(bbs[sel]):
                        image = Image.open(os.path.join(img_path, f'{i * batch_size + idx // 4:05}_{idx % 4}.png'))
                        width, height, center_x, center_y = bb.tolist()
                        center_x = (center_x + 1.0) / 2.0 * 128
                        center_y = (center_y + 1.0) / 2.0 * 128
                        bb = (int(center_x - width * 128),
                              int(center_y - height * 128),
                              int(center_x + width * 128),
                              int(center_y + height * 128))
                        try:
                            cropped = image.crop(bb)
                            cropped.save(f'{self.relevant_object_hover_path}/img/'
                                         f'gs{global_step:06}_{i * batch_size + idx // 4:05}_{idx % 4}_obj{obj_idx}.png')
                        except:
                            image.save(f'{self.relevant_object_hover_path}/img/'
                                       f'gs{global_step:06}_{i * batch_size + idx // 4:05}_{idx % 4}_obj{obj_idx}.png')
                        new_image_path = f'gs{global_step:06}_{i * batch_size + idx // 4:05}_{idx % 4}_obj{obj_idx}.png'
                        image_refs.append(new_image_path)
            # (N, 32)
            boxes_batch = convert_to_boxes(z_where, z_pres, z_pres_prob, with_conf=True)
            z_whats.extend(z_what[z_pres])
            for j in range(len(z_pres_prob) // 4):
                datapoint_encs = []
                for k in range(4):
                    z_wr, z_pr, z_wt = z_where[j * 4 + k], z_pres_prob[j * 4 + k], z_what[j * 4 + k]
                    z_pr = z_pr.squeeze() > 0.5
                    datapoint_encs.append(torch.cat([z_wr[z_pr], z_wt[z_pr]], dim=1))
                z_encs.append(datapoint_encs)
            all_labels.extend(dataset.get_labels(i * batch_size, (i + 1) * batch_size, boxes_batch))
            all_labels_moving.extend(
                dataset.get_labels_moving(i * batch_size, (i + 1) * batch_size, boxes_batch))
        args = {'type': 'classify', 'method': 'kmeans', 'indices': None, 'dim': 2, 'folder': 'validation',
                'edgecolors': False}
        if z_whats:
            z_whats = torch.stack(z_whats).detach().cpu()
            all_labels_relevant_idx, all_labels_relevant = dataset.to_relevant(all_labels_moving)
            z_whats_relevant = z_whats[flatten(all_labels_relevant_idx)]
            all_objects = evaluate_z_what(args, z_whats, flatten(all_labels), len(z_whats), cfg, title="all")
            moving_objects = evaluate_z_what(args, z_whats, flatten(all_labels_moving), len(z_whats), cfg,
                                             title="moving")
            relevant_objects = evaluate_z_what(args, z_whats_relevant, flatten(all_labels_relevant), len(z_whats), cfg,
                                               title="relevant")
            z_encs_relevant = [[enc[rel_idx] for enc, rel_idx in zip(enc_seq, rel_seq)]
                               for enc_seq, rel_seq in zip(z_encs, all_labels_relevant_idx)]
            bayes_accuracy = classify_encodings(cfg, z_encs_relevant, all_labels_relevant)
            relevant_objects[2]['bayes_accuracy'] = bayes_accuracy
            if cfg.save_relevant_objects:
                with open(f'{self.relevant_object_hover_path}/relevant_objects_{global_step:06}.pkl',
                          'wb') as output_file:
                    relevant_objects_data = {
                        'z_what': z_whats_relevant,
                        'labels': all_labels_relevant,
                        'image_refs': [image_refs[idx] for idx, yes in enumerate(all_labels_relevant_idx) if yes]
                    }
                    pickle.dump(relevant_objects_data, output_file, pickle.DEFAULT_PROTOCOL)
        else:
            all_objects = evaluate_z_what(args, None, None, None, cfg, title="all")
            moving_objects = evaluate_z_what(args, None, None, None, cfg, title="moving")
            relevant_objects = evaluate_z_what(args, None, None, None, cfg, title="relevant")
        return {'all': all_objects, 'moving': moving_objects, 'relevant': relevant_objects}

    @torch.no_grad()
    def eval_ap_and_acc(self, logs, dataset, bb_path, iou_thresholds=None):
        """
        Evaluate average precision and accuracy
        :param logs: the model output
        :param dataset: the dataset for accessing label information
        :param bb_path: directory containing the gt bounding boxes.
        :param iou_thresholds:
        :return ap: a list of average precisions, corresponding to each iou_thresholds
        """
        batch_size = eval_cfg.train.batch_size
        num_samples = eval_cfg.train.num_samples.ap
        print('Computing error rates, counts and APs...')
        if iou_thresholds is None:
            iou_thresholds = np.linspace(0.1, 0.9, 9)
        boxes_gt_types = ['all', 'moving', 'relevant']
        indices = list(range(num_samples))
        boxes_gts = {k: v for k, v in zip(boxes_gt_types, read_boxes(bb_path, indices=indices))}
        boxes_pred = []
        boxes_relevant = []

        rgb_folder_src = f"../aiml_atari_data_mid/rgb/Pong-v0/validation"
        rgb_folder = f"../aiml_atari_data2/with_bounding_boxes/Pong-v0/sample"
        num_batches = eval_cfg.train.num_samples.cluster // eval_cfg.train.batch_size

        for img in logs[:num_batches]:
            z_where, z_pres_prob, z_what = img['z_where'], img['z_pres_prob'], img['z_what']
            z_where = z_where.detach().cpu()
            z_pres_prob = z_pres_prob.detach().cpu().squeeze()
            z_pres = z_pres_prob > 0.5
            boxes_batch = convert_to_boxes(z_where, z_pres, z_pres_prob, with_conf=True)
            boxes_relevant.extend(dataset.filter_relevant_boxes(boxes_batch, boxes_gts['all']))
            boxes_pred.extend(boxes_batch)

        # print('Drawing bounding boxes for eval...')
        # for i in range(4):
        #     for idx, pred, rel, gt, gt_m, gt_r in zip(indices, boxes_pred[i::4], boxes_relevant[i::4], *(gt[i::4] for gt in boxes_gts.values())):
        #         if len(pred) == len(rel):
        #             continue
        #         pil_img = Image.open(f'{rgb_folder_src}/{idx:05}_{i}.png').convert('RGB')
        #         pil_img = pil_img.resize((128, 128), PIL.Image.BILINEAR)
        #         image = np.array(pil_img)
        #         torch_img = torch.from_numpy(image).permute(2, 1, 0)
        #         pred_tensor = torch.FloatTensor(pred) * 128
        #         pred_tensor = torch.index_select(pred_tensor, 1, torch.LongTensor([0, 2, 1, 3]))
        #         rel_tensor = torch.FloatTensor(rel) * 128
        #         rel_tensor = torch.index_select(rel_tensor, 1, torch.LongTensor([0, 2, 1, 3]))
        #         gt_tensor = torch.FloatTensor(gt) * 128
        #         gt_tensor = torch.index_select(gt_tensor, 1, torch.LongTensor([0, 2, 1, 3]))
        #         gt_m_tensor = torch.FloatTensor(gt_m) * 128
        #         gt_m_tensor = torch.index_select(gt_m_tensor, 1, torch.LongTensor([0, 2, 1, 3]))
        #         gt_r_tensor = torch.FloatTensor(gt_r) * 128
        #         gt_r_tensor = torch.index_select(gt_r_tensor, 1, torch.LongTensor([0, 2, 1, 3]))
        #         bb_img = torch_img
        #         bb_img = draw_bb(torch_img, gt_tensor, colors=["red"] * len(gt_tensor))
        #         # bb_img = draw_bb(bb_img, gt_m_tensor, colors=["blue"] * len(gt_m_tensor))
        #         # bb_img = draw_bb(bb_img, gt_r_tensor, colors=["yellow"] * len(gt_r_tensor))
        #         bb_img = draw_bb(bb_img, pred_tensor, colors=["green"] * len(pred_tensor))
        #         # bb_img = draw_bb(bb_img, rel_tensor, colors=["white"] * len(rel_tensor))
        #         bb_img = Image.fromarray(bb_img.permute(2, 1, 0).numpy())
        #         bb_img.save(f'{rgb_folder}/gt_moving_p{idx:05}_{i}.png')
        #         print(f'{rgb_folder}/gt_moving_{idx:05}.png')

        result = {}
        for gt_name, gt in boxes_gts.items():
            # Four numbers
            boxes = boxes_pred if gt_name != "relevant" else boxes_relevant
            error_rate, perfect, overcount, undercount = compute_counts(boxes, gt)
            accuracy = perfect / (perfect + overcount + undercount)
            result[f'error_rate_{gt_name}'] = error_rate
            result[f'perfect_{gt_name}'] = perfect
            result[f'accuracy_{gt_name}'] = accuracy
            result[f'overcount_{gt_name}'] = overcount
            result[f'undercount_{gt_name}'] = undercount
            result[f'iou_thresholds_{gt_name}'] = iou_thresholds
            # A list of length 9 and P/R from low IOU level = 0.2
            aps = compute_ap(boxes, gt, iou_thresholds)
            precision, recall = compute_prec_rec(boxes, gt)
            result[f'APs_{gt_name}'] = aps
            result[f'precision_{gt_name}'] = precision
            result[f'recall_{gt_name}'] = recall
        return result

    def save_to_json(self, result_dict, json_path, info):
        """
        Save evaluation results to json file

        :param result_dict: a dictionary
        :param json_path: checkpointdir
        :param info: any other thing you want to save
        :return:
        """
        from collections import OrderedDict
        import json
        from datetime import datetime
        tosave = OrderedDict([
            ('date', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ('info', info),
        ])
        for metric in ['APs', 'accuracy', 'error_rate', 'iou_thresholds', 'overcount', 'perfect', 'precision', 'recall', 'undercount']:
            if hasattr(result_dict[f'{metric}_all'], '__iter__'):
                tosave[metric] = list(result_dict[f'{metric}_all'])
                tosave[f'{metric}_avg'] = np.mean(result_dict[f'{metric}_all'])
                tosave[f'{metric}_relevant'] = list(result_dict[f'{metric}_relevant'])
                tosave[f'{metric}_relevant_avg'] = np.mean(result_dict[f'{metric}_relevant'])
                tosave[f'{metric}_moving'] = list(result_dict[f'{metric}_moving'])
                tosave[f'{metric}_moving_avg'] = np.mean(result_dict[f'{metric}_moving'])
            else:
                tosave[metric] = result_dict[f'{metric}_all']
                tosave[f'{metric}_relevant'] = result_dict[f'{metric}_relevant']
                tosave[f'{metric}_moving'] = result_dict[f'{metric}_moving']
        with open(json_path, 'w') as f:
            json.dump(tosave, f, indent=2)

        print(f'Results have been saved to {json_path}.')

    def print_result(self, result_dict, files):
        for suffix in ['all', 'relevant', 'moving']:
            APs = result_dict[f'APs_{suffix}']
            iou_thresholds = result_dict[f'iou_thresholds_{suffix}']
            accuracy = result_dict[f'accuracy_{suffix}']
            perfect = result_dict[f'perfect_{suffix}']
            overcount = result_dict[f'overcount_{suffix}']
            undercount = result_dict[f'undercount_{suffix}']
            error_rate = result_dict[f'error_rate_{suffix}']
            for file in files:
                print('\n' + '-' * 10 + f'metrics on {suffix} data points' + '-' * 10, file=file)
                print('{:^15} {:^15}'.format('IoU threshold', 'AP'), file=file)
                print('{:15} {:15}'.format('-' * 15, '-' * 15), file=file)
                for thres, ap in zip(iou_thresholds, APs):
                    print('{:<15.2} {:<15.4}'.format(thres, ap), file=file)
                print('{:15} {:<15.4}'.format('Average:', np.mean(APs)), file=file)
                print('{:15} {:15}'.format('-' * 15, '-' * 15), file=file)

                print('{:15} {:<15}'.format('Perfect:', perfect), file=file)
                print('{:15} {:<15}'.format('Overcount:', overcount), file=file)
                print('{:15} {:<15}'.format('Undercount:', undercount), file=file)
                print('{:15} {:<15.4}'.format('Accuracy:', accuracy), file=file)
                print('{:15} {:<15.4}'.format('Error rate:', error_rate), file=file)
                print('{:15} {:15}'.format('-' * 15, '-' * 15), file=file)

    # def save_best(self, evaldir, metric_name, value, checkpoint, checkpointer, min_is_better):
    #     metric_file = os.path.join(evaldir, f'best_{metric_name}.json')
    #     checkpoint_file = os.path.join(evaldir, f'best_{metric_name}.pth')
    #
    #     now = datetime.now()
    #     log = {
    #         'name': metric_name,
    #         'value': float(value),
    #         'date': now.strftime("%Y-%m-%d %H:%M:%S"),
    #         'global_step': checkpoint[-1]
    #     }
    #
    #     if not os.path.exists(metric_file):
    #         dump = True
    #     else:
    #         with open(metric_file, 'r') as f:
    #             previous_best = json.load(f)
    #         if not math.isfinite(log['value']):
    #             dump = True
    #         elif (min_is_better and log['value'] < previous_best['value']) or (
    #                 not min_is_better and log['value'] > previous_best['value']):
    #             dump = True
    #         else:
    #             dump = False
    #     if dump:
    #         with open(metric_file, 'w') as f:
    #             json.dump(log, f)
    #         checkpointer.save(checkpoint_file, *checkpoint)
