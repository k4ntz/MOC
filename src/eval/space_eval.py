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
from .ap import read_boxes, convert_to_boxes, compute_ap, compute_counts
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from post_eval import evaluate_z_what
import PIL
from torchvision.utils import draw_bounding_boxes as draw_bb
import os
import pprint


class SpaceEval:
    def __init__(self):
        self.eval_file = None
        self.eval_file_path = None
        self.first_eval = True

    @torch.no_grad()
    def test_eval(self, model, testset, bb_path, device, evaldir, info):
        losses, logs = self.apply_model(testset, device, model)
        result_dict = self.eval_ap_and_acc(logs, testset, bb_path)
        os.makedirs(evaldir, exist_ok=True)
        path = osp.join(evaldir, 'results_{}.json'.format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
        self.save_to_json(result_dict, path, info)
        self.print_result(result_dict, [sys.stdout, open('./results.txt', 'w')])

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
        if 'mse' in eval_cfg.train.metrics:
            columns.append('mse')
        if 'ap' in eval_cfg.train.metrics:
            for class_name in ['all', 'moving', 'relevant']:
                for iou_t in np.linspace(0.05, 0.95, 19):
                    columns.append(f'{class_name}_ap_{iou_t:.2f}')
                columns.append(f'{class_name}_accuracy')
                columns.append(f'{class_name}_perfect')
                columns.append(f'{class_name}_overcount')
                columns.append(f'{class_name}_undercount')
                columns.append(f'{class_name}_error_rate')

        with open(self.eval_file_path, "w") as file:
            file.write(";".join(columns))
            file.write("\n")


    @torch.no_grad()
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
            self.eval_file_path = f'{cfg.logdir}/{cfg.exp_name}{cfg.seed}/metrics.csv'
            if os.path.exists(self.eval_file_path):
                os.remove(self.eval_file_path)
            self.write_header()

        losses, logs = self.apply_model(valset, device, model)
        with open(self.eval_file_path, "a") as self.eval_file:
            self.write_metric(None, None, global_step, global_step, use_writer=False)
            if 'cluster' in eval_cfg.train.metrics:
                results = self.train_eval_clustering(logs, valset, writer, global_step, cfg)
                print("Cluster Result:", results)
                checkpointer.save_best('mutual_information_all', results['all'][0]['adjusted_mutual_info_score'],
                                       checkpoint, min_is_better=False)
                checkpointer.save_best('mutual_information_relevant',
                                       results['relevant'][0]['adjusted_mutual_info_score'],
                                       checkpoint, min_is_better=False)
            if 'mse' in eval_cfg.train.metrics:
                mse = self.train_eval_mse(logs, losses, writer, global_step)
                print("MSE result: ", mse)
            if 'ap' in eval_cfg.train.metrics:
                results = self.train_eval_ap_and_acc(logs, valset, bb_path, writer, global_step)
                APs = results['APs_relevant']
                checkpointer.save_best('error_rate_relevant', APs[len(APs) // 2], checkpoint, min_is_better=True)
                checkpointer.save_best('error_rate_relevant', results['error_rate_relevant'], checkpoint, min_is_better=True)
                results = {k2: v2[len(v2) // 2] if isinstance(v2, list) or isinstance(v2, np.ndarray) else v2 for k2, v2, in
                           results.items()}
                pp = pprint.PrettyPrinter(depth=2)
                print("AP Result:")
                pp.pprint(results)
            self.eval_file.write("\n")

    @torch.no_grad()
    def apply_model(self, dataset, device, model):
        print('Applying the model for evaluation...')
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
            for i, imgs in enumerate(dataloader):
                imgs = imgs.to(device)
                loss, log = model(imgs, global_step=100000000)
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

            for ap in APs:
                self.write_metric(None, 'ignored', ap, global_step, use_writer=False)
            for ap, thres in zip(APs[1::4], iou_thresholds[1::4]):
                writer.add_scalar(f'val_aps_{class_name}/ap_{thres}', ap, global_step)
            writer.add_scalar(f'{class_name}/ap_avg_0.5', APs[len(APs) // 2], global_step)
            writer.add_scalar(f'{class_name}/ap_avg_up', np.mean(APs[len(APs) // 2:]), global_step)
            writer.add_scalar(f'{class_name}/ap_avg', np.mean(APs), global_step)
            self.write_metric(writer, f'{class_name}/accuracy', accuracy, global_step)
            self.write_metric(writer, f'{class_name}/perfect', perfect, global_step)
            self.write_metric(writer, f'{class_name}/overcount', overcount, global_step)
            self.write_metric(writer, f'{class_name}/undercount', undercount, global_step)
            self.write_metric(writer, f'{class_name}/error_rate', error_rate, global_step, make_sep=class_name != 'relevant')
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
            B = eval_cfg.train.batch_size
            for b in range(B):
                metric_logger.update(
                    mse=torch.mean(torch.tensor([log['space_log'][i]['mse'][b] for i in range(4)]))
                )
            metric_logger.update(loss=loss.mean())
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
        results = self.eval_clustering(logs, valset, cfg)
        for name, (result_dict, img_path, few_shot_accuracy) in results.items():
            writer.add_image(f'Clustering PCA {name.title()}', np.array(Image.open(img_path)), global_step,
                             dataformats='HWC')
            for score in few_shot_accuracy:
                self.write_metric(writer, f'{name}/{score}', few_shot_accuracy[score], global_step)
            for score in result_dict:
                self.write_metric(writer, f'{name}/{score}', result_dict[score], global_step)
        return results

    @torch.no_grad()
    def eval_clustering(self, logs, dataset, cfg):
        """
        Evaluate clustering metrics

        :param logs: results from applying the model
        :param dataset: dataset
        :param cfg: config
        :return metrics: for all classes of evaluation many metrics describing the clustering,
            based on different ground truths
        """
        z_encs = []
        all_labels = []
        all_labels_moving = []
        batch_size = eval_cfg.train.batch_size
        num_batches = eval_cfg.train.num_samples.cluster // batch_size
        for i, log in enumerate(logs[:num_batches]):
            for j, img in enumerate(log['space_log']):
                z_where, z_pres_prob, z_what = img['z_where'], img['z_pres_prob'], img['z_what']
                z_pres_prob = z_pres_prob.squeeze()
                z_pres = z_pres_prob > 0.5
                # (N, 32)
                z_enc = z_what[z_pres]
                boxes_batch = convert_to_boxes(z_where, z_pres, z_pres_prob, with_conf=True)
                z_encs.extend(z_enc)
                all_labels.extend(dataset.get_labels(range(i * batch_size, (i + 1) * batch_size), j, boxes_batch))
                all_labels_moving.extend(
                    dataset.get_labels_moving(range(i * batch_size, (i + 1) * batch_size), j, boxes_batch))
        args = {'type': 'classify', 'method': 'kmeans', 'indices': None, 'dim': 2, 'folder': 'validation',
                'edgecolors': False}
        z_encs = torch.stack(z_encs).detach().cpu()
        all_labels = torch.cat(all_labels).detach().cpu()
        all_labels_moving = torch.cat(all_labels_moving).detach().cpu()
        all_labels_relevant_idx = dataset.to_relevant(all_labels_moving)
        all_labels_relevant = all_labels_moving[all_labels_relevant_idx]
        z_encs_relevant = z_encs[all_labels_relevant_idx]
        all_objects = evaluate_z_what(args, z_encs, all_labels, len(z_encs), cfg, title="all")
        moving_objects = evaluate_z_what(args, z_encs, all_labels_moving, len(z_encs), cfg, title="moving")
        relevant_objects = evaluate_z_what(args, z_encs_relevant, all_labels_relevant, len(z_encs), cfg, title="relevant")
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
            iou_thresholds = np.linspace(0.05, 0.95, 19)
        boxes_gt_types = ['all', 'moving', 'relevant']
        indices = [k * batch_size * 4 + i * 4 + j + dataset.flow for k in range(num_samples // batch_size) for j in
                   range(4) for i in range(batch_size)]
        boxes_gts = {k: v for k, v in zip(boxes_gt_types, read_boxes(bb_path, 128, indices))}
        boxes_pred = []
        boxes_relevant = []

        rgb_folder_src = f"../aiml_atari_data/rgb/MsPacman-v0/validation"
        rgb_folder = f"../aiml_atari_data/with_bounding_boxes/MsPacman-v0/validation"
        num_batches = eval_cfg.train.num_samples.cluster // eval_cfg.train.batch_size
        for log in logs[:num_batches]:
            for j, img in enumerate(log['space_log']):
                z_where, z_pres_prob, z_what = img['z_where'], img['z_pres_prob'], img['z_what']
                z_where = z_where.detach().cpu()
                z_pres_prob = z_pres_prob.detach().cpu().squeeze()
                z_pres = z_pres_prob > 0.5
                boxes_batch = convert_to_boxes(z_where, z_pres, z_pres_prob, with_conf=True)
                boxes_relevant.extend(dataset.filter_relevant_boxes(boxes_batch))
                boxes_pred.extend(boxes_batch)

        # print('Drawing bounding boxes for eval...')
        # for idx, pred, rel, gt, gt_m, gt_r in zip(indices, boxes_pred, boxes_relevant, *boxes_gts.values()):
        #     pil_img = Image.open(f'{rgb_folder_src}/{idx:05}.png', ).convert('RGB')
        #     pil_img = pil_img.resize((128, 128), PIL.Image.BILINEAR)
        #     image = np.array(pil_img)
        #     torch_img = torch.from_numpy(image).permute(2, 1, 0)
        #     pred_tensor = torch.FloatTensor(pred) * 128
        #     pred_tensor = torch.index_select(pred_tensor, 1, torch.LongTensor([0, 2, 1, 3]))
        #     rel_tensor = torch.FloatTensor(rel) * 128
        #     rel_tensor = torch.index_select(rel_tensor, 1, torch.LongTensor([0, 2, 1, 3]))
        #     gt_tensor = torch.FloatTensor(gt) * 128
        #     gt_tensor = torch.index_select(gt_tensor, 1, torch.LongTensor([0, 2, 1, 3]))
        #     gt_m_tensor = torch.FloatTensor(gt_m) * 128
        #     gt_m_tensor = torch.index_select(gt_m_tensor, 1, torch.LongTensor([0, 2, 1, 3]))
        #     gt_r_tensor = torch.FloatTensor(gt_r) * 128
        #     gt_r_tensor = torch.index_select(gt_r_tensor, 1, torch.LongTensor([0, 2, 1, 3]))
        #     bb_img = draw_bb(torch_img, gt_tensor, colors='red')
        #     bb_img = draw_bb(bb_img, gt_m_tensor, colors='blue')
        #     bb_img = draw_bb(bb_img, gt_r_tensor, colors='yellow')
        #     bb_img = draw_bb(bb_img, pred_tensor, colors='green')
        #     bb_img = draw_bb(bb_img, rel_tensor, colors='white')
        #     bb_img = Image.fromarray(bb_img.permute(2, 1, 0).numpy())
        #     bb_img.save(f'{rgb_folder}/temp_from_eval_objects_{idx:05}.png')
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
            # A list of length 19
            result[f'APs_{gt_name}'] = compute_ap(boxes, gt, iou_thresholds)
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
            ('APs', list(result_dict['APs'])),
            ('iou_thresholds', list(result_dict['iou_thresholds'])),
            ('AP average', np.mean(result_dict['APs'])),
            ('error_rate', result_dict['error_rate']),
            ('accuracy', result_dict['accuracy']),
            ('perfect', result_dict['perfect']),
            ('undercount', result_dict['undercount']),
            ('overcount', result_dict['overcount']),
        ])
        with open(json_path, 'w') as f:
            json.dump(tosave, f, indent=2)

        print(f'Results have been saved to {json_path}.')

    def print_result(self, result_dict, files):
        APs = result_dict['APs']
        iou_thresholds = result_dict['iou_thresholds']
        accuracy = result_dict['accuracy']
        perfect = result_dict['perfect']
        overcount = result_dict['overcount']
        undercount = result_dict['undercount']
        error_rate = result_dict['error_rate']
        for file in files:
            print('-' * 30, file=file)
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
