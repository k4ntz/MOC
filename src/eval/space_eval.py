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
from post_eval import evaluate_z_what, get_labels_validation

class SpaceEval():
    def __init__(self):
        pass

    @torch.no_grad()
    def test_eval(self, model, testset, bb_path, device, evaldir, info):
        result_dict = self.eval_ap_and_acc(
            model, testset, bb_path, eval_cfg.test.batch_size, eval_cfg.test.num_workers,
            device, num_samples=None
        )
        os.makedirs(evaldir, exist_ok=True)
        path = osp.join(evaldir, 'results_{}.json'.format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
        self.save_to_json(result_dict, path, info)
        self.print_result(result_dict, [sys.stdout, open('./results.txt', 'w')])
        # APs = result_dict['APs']
        # iou_thresholds = result_dict['iou_thresholds']
        # accuracy = result_dict['accuracy']
        # perfect = result_dict['perfect']
        # overcount = result_dict['O']
        # undercount = result_dict['undercount']
        # error_rate = result_dict['error_rate']
        

    @torch.no_grad()
    def train_eval(self, model, valset, bb_path, writer, global_step, device, checkpoint, checkpointer, cfg):
        """
        Evaluation during training. This includes:
            - mse evaluated on validation set
            - ap and accuracy evaluated on validation set
            - cluster metrics evaluated on validation set
        :return:
        """
        if 'cluster' in eval_cfg.train.metrics:
            results = self.train_eval_clustering(model, valset, bb_path, writer, global_step, device, cfg)
            print("Cluster result: ", results)
        if 'mse' in eval_cfg.train.metrics:
            mse = self.train_eval_mse(model, valset, writer, global_step, device)
            print("MSE result: ", mse)
        if 'ap' in eval_cfg.train.metrics:
            results = self.train_eval_ap_and_acc(model, valset, bb_path, writer, global_step, device)
            print("AP result: ", results['APs'], results['accuracy'])
            checkpointer.save_best('ap_dot5', results['APs'][0], checkpoint, min_is_better=False)
            checkpointer.save_best('ap_avg', np.mean(results['APs']), checkpoint, min_is_better=False)
            checkpointer.save_best('error_rate', results['error_rate'], checkpoint, min_is_better=True)

        
    @torch.no_grad()
    def train_eval_ap_and_acc(self, model, valset, bb_path, writer: SummaryWriter, global_step, device):
        """
        Evaluate ap and accuracy during training
        
        :return: result_dict
        """
        result_dict = self.eval_ap_and_acc(
            model, valset, bb_path, eval_cfg.train.batch_size, eval_cfg.train.num_workers,
            device, num_samples=eval_cfg.train.num_samples.ap
        )
        APs = result_dict['APs']
        iou_thresholds = result_dict['iou_thresholds']
        accuracy = result_dict['accuracy']
        perfect = result_dict['perfect']
        overcount = result_dict['overcount']
        undercount = result_dict['undercount']
        error_rate = result_dict['error_rate']
        
        for ap, thres in zip(APs, iou_thresholds):
            writer.add_scalar(f'val_aps/ap_{thres}', ap, global_step)
        writer.add_scalar(f'val/ap_avg', np.mean(APs), global_step)
        writer.add_scalar('val/accuracy', accuracy, global_step)
        writer.add_scalar('val/perfect', perfect, global_step)
        writer.add_scalar('val/overcount', overcount, global_step)
        writer.add_scalar('val/undercount', undercount, global_step)
        writer.add_scalar('val/error_rate', error_rate, global_step)
        for z_pres_thresh in result_dict['z_pres_thresholds']:
            if f'accuracy_{z_pres_thresh}' not in result_dict:
                continue
            writer.add_scalar(f'val/accuracy_{z_pres_thresh}', result_dict[f'accuracy_{z_pres_thresh}'], global_step)
            writer.add_scalar(f'val/perfect_{z_pres_thresh}', result_dict[f'perfect_{z_pres_thresh}'], global_step)
            writer.add_scalar(f'val/overcount_{z_pres_thresh}', result_dict[f'overcount_{z_pres_thresh}'], global_step)
            writer.add_scalar(f'val/undercount_{z_pres_thresh}', result_dict[f'undercount_{z_pres_thresh}'], global_step)
            writer.add_scalar(f'val/error_rate_{z_pres_thresh}', result_dict[f'error_rate_{z_pres_thresh}'], global_step)
        return result_dict

    @torch.no_grad()
    def train_eval_mse(self, model, valset, writer, global_step, device):
        """
        Evaluate MSE during training
        """
        num_samples = eval_cfg.train.num_samples.mse
        batch_size = eval_cfg.train.batch_size
        num_workers = eval_cfg.train.num_workers
        
        model.eval()
        valset = Subset(valset, indices=range(num_samples))
        dataloader = DataLoader(valset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
        metric_logger = MetricLogger()
    
        print(f'Evaluating MSE using {num_samples} samples.')
        # with tqdm(total=len(dataloader)) as pbar:
        for batch_idx, sample in enumerate(dataloader):
            imgs = sample.to(device)
            loss, log = model(imgs, global_step)
            B = imgs.size(0)
            for b in range(B):
                metric_logger.update(
                    mse=torch.mean([log['space_log'][i]['mse'][b] for i in range(4)])
                )
            metric_logger.update(loss=loss.mean())

        assert metric_logger['mse'].count == num_samples
        # Add last log
        # log.update([(k, torch.tensor(v.global_avg)) for k, v in metric_logger.values.items()])
        mse = metric_logger['mse'].global_avg
        writer.add_scalar(f'val/mse', mse, global_step=global_step)
    
        model.train()
        
        return mse

    @torch.no_grad()
    def train_eval_clustering(self, model, valset, bb_path, writer: SummaryWriter, global_step, device, cfg):
        """
        Evaluate clustering during training
        :return: result_dict
        """
        print('Computing clustering and few-shot linear classifiers...')

        result_dict, img_path, few_shot_accuracy = self.eval_clustering(
            model, valset, bb_path, eval_cfg.train.batch_size, eval_cfg.train.num_workers,
            device, cfg, num_samples=eval_cfg.train.num_samples.cluster
        )

        writer.add_image('Clustering PCA', np.array(Image.open(img_path)), global_step, dataformats='HWC')
        for score in few_shot_accuracy:
            writer.add_scalar(f'val/{score}', few_shot_accuracy[score], global_step)
        for score in result_dict:
            writer.add_scalar(f'val/{score}', result_dict[score], global_step)
        return result_dict

    def eval_clustering(
            self,
            model,
            dataset,
            bb_path,
            batch_size,
            num_workers,
            device,
            cfg,
            num_samples=None
    ):
        """
        Evaluate clustering metrics

        :param model: Space
        :param dataset: dataset
        :param bb_path: directory containing the gt bounding boxes.
        :param batch_size: batch size
        :param num_workers: num_workers
        :param device: device
        :param num_samples: number of samples for evaluating it. If None use all samples
        :return metrics: a list of metrics describing the cluster, based on the ground truth
        :return pca of the data involved
        """

        from tqdm import tqdm

        model.eval()

        if num_samples is None:
            num_samples = len(dataset)
        dataset = Subset(dataset, indices=range(num_samples))
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

        z_whats = []
        all_labels = []
        model.eval()
        with torch.no_grad():
            # pbar = tqdm(total=len(dataloader))
            for i, imgs in enumerate(dataloader):
                imgs = imgs.to(device)
                loss, log = model(imgs, global_step=100000000)
                img = log['space_log'][0]
                z_where, z_pres_prob, z_what = img['z_where'], img['z_pres_prob'], img['z_what']
                z_where = z_where.detach().cpu()
                z_pres_prob = z_pres_prob.detach().cpu().squeeze()
                z_pres = z_pres_prob > 0.5
                z_what_pres = z_what[z_pres]
                boxes_batch = convert_to_boxes(z_where, z_pres, z_pres_prob, with_conf=True)
                z_whats.extend(z_what_pres)
                all_labels.extend(get_labels_validation(range(i * batch_size, (i + 1) * batch_size), boxes_batch))
                # pbar.update(1)
        model.train()
        args = {'type': 'classify', 'method': 'kmeans', 'indices': None, 'dim': 2, 'folder': 'validation'}
        return evaluate_z_what(args, torch.stack(z_whats).detach().cpu(), torch.cat(all_labels).detach().cpu(), len(z_whats), cfg)

    def eval_ap_and_acc(
            self,
            model,
            dataset,
            bb_path, 
            batch_size,
            num_workers,
            device,
            num_samples=None,
            iou_thresholds=None,
    ):
        """
        Evaluate average precision and accuracy
        
        :param model: Space
        :param dataset: dataset
        :param bb_path: directory containing the gt bounding boxes.
        :param batch_size: batch size
        :param num_workers: num_workers
        :param device: device
        :param num_samples: number of samples for evaluating it. If None use all samples
        :param iou_thresholds:
        :return ap: a list of average precisions, corresponding to each iou_thresholds
        """
        
        from tqdm import tqdm
        print('Computing error rates, counts and APs...')
        model.eval()
        
        if num_samples is None:
            num_samples = len(dataset)
        dataset = Subset(dataset, indices=range(num_samples))
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        
        if iou_thresholds is None:
            iou_thresholds = np.linspace(0.5, 0.95, 10)
        z_pres_thresholds = [0.5]

        boxes_gt = read_boxes(bb_path, 128)
        boxes_preds = [[] for _ in z_pres_thresholds]
        result = {
            'iou_thresholds': iou_thresholds,
            'z_pres_thresholds': z_pres_thresholds,
        }
        rgb_folder_src = f"../aiml_atari_data/rgb/MsPacman-v0/validation"
        rgb_folder = f"../aiml_atari_data/with_bounding_boxes/MsPacman-v0/validation"
        model.eval()
        with torch.no_grad():
            # pbar = tqdm(total=len(dataloader))
            for i, imgs in enumerate(dataloader):
                imgs = imgs.to(device)
            
                loss, log = model(imgs, global_step=100000000)
                img = log['space_log'][0]
                # (B, N, 4), (B, N, 1), (B, N, 1)
                z_where, z_pres_prob = img['z_where'], img['z_pres_prob']
                # (B, N, 4), (B, N), (B, N)
                z_where = z_where.detach().cpu()
                z_pres_prob = z_pres_prob.detach().cpu().squeeze()
                for pred_idx, prob_th_str in enumerate(z_pres_thresholds):
                    z_pres = z_pres_prob > prob_th_str  # try other values
                    boxes_batch = convert_to_boxes(z_where, z_pres, z_pres_prob, with_conf=True)
                    boxes_preds[pred_idx].extend(boxes_batch)
                # pbar.update(1)

            # print('Drawing bounding boxes for eval...')
            # for idx, (pred, gt) in enumerate(zip(boxes_pred, boxes_gt)):
            #     pil_img = Image.open(f'{rgb_folder_src}/{idx:05}.png', ).convert('RGB')
            #     pil_img = pil_img.resize((128, 128), PIL.Image.BILINEAR)
            #     image = np.array(pil_img)
            #     torch_img = torch.from_numpy(image).permute(2, 1, 0)
            #     pred_tensor = torch.FloatTensor(pred) * 128
            #     pred_tensor = torch.index_select(pred_tensor, 1, torch.LongTensor([0, 2, 1, 3]))
            #     gt_tensor = torch.FloatTensor(gt) * 128
            #     gt_tensor = torch.index_select(gt_tensor, 1, torch.LongTensor([0, 2, 1, 3]))
            #     bb_img = draw_bb(torch_img, gt_tensor, colors=['red']*50)
            #     bb_img = draw_bb(bb_img, pred_tensor, colors=['green']*50)
            #     result = Image.fromarray(bb_img.permute(2, 1, 0).numpy())
            #     result.save(f'{rgb_folder}/temp_from_eval_objects_{idx:05}.png')



            # Four numbers
            for pred_idx, prob_thresh in enumerate(z_pres_thresholds):
                error_rate, perfect, overcount, undercount = compute_counts(boxes_preds[pred_idx], boxes_gt)
                accuracy = perfect / (perfect + overcount + undercount)
                prob_th_str = f'_{prob_thresh}' if prob_thresh != 0.5 else ''
                result[f'error_rate{prob_th_str}'] = error_rate
                result[f'perfect{prob_th_str}'] = perfect
                result[f'accuracy{prob_th_str}'] = accuracy
                result[f'overcount{prob_th_str}'] = overcount
                result[f'undercount{prob_th_str}'] = undercount
            # A list of length 10
            result['APs'] = compute_ap(boxes_preds[0], boxes_gt, iou_thresholds)
    
        model.train()
        
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
