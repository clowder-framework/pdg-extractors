import logging
import json
import matplotlib.pyplot as plt
from detectron2.engine.train_loop import HookBase
import torch
import numpy as np
from utils.metrics import SegmentationMetrics


class InstanceSegFigureHook(HookBase):
    def __init__(self, output_dir, eval_period):
        self._output_dir = output_dir
        self._period = eval_period
        self._logger = logging.getLogger("__name__")

    def _do_instance_seg_figure(self):
        metric_file = "{}/metrics.json".format(self._output_dir)

        # Every line in the metrics.json file is its own json object.
        with open(metric_file, "r") as f:
            lines = [line for line in f.readlines()]

        if len(lines) == 0:
            return

        train_loss = []
        val_loss = []
        bbox_map = []
        seg_map = []
        itr_train_loss = []
        itr_val_loss = []
        itr_bbox = []
        itr_seg = []
        for line in lines:
            data = json.loads(line)
            itr = data["iteration"]
            if "total_loss" in data:
                train_loss.append(data["total_loss"])
                itr_train_loss.append(itr)
            if "validation_loss" in data:
                val_loss.append(data["validation_loss"])
                itr_val_loss.append(itr)
            if "bbox/AP50" in data:
                bbox_map.append(data["bbox/AP50"])
                itr_bbox.append(itr)
            if "segm/AP50" in data:
                seg_map.append(data["segm/AP50"])
                itr_seg.append(itr)

        figure, axis = plt.subplots(1, 2)

        # plot loss figure
        axis[0].plot(itr_train_loss, train_loss, label="loss")
        axis[0].plot(itr_val_loss, val_loss, label="val_loss")
        axis[0].legend(loc="upper right")
        axis[0].set_title("Loss over time")

        # plot map figure
        axis[1].plot(itr_bbox, bbox_map, label="bbox_mAP")
        axis[1].plot(itr_seg, seg_map, label="segm_mAP")
        axis[1].legend(loc="lower right")
        axis[1].set_title("BBox and Segm mAP over time")
        axis[1].scatter(
            (bbox_map.index(max(bbox_map)) + 1) * self._period,
            max(bbox_map),
            c="r",
            label="best_bbox_map50: {:.2f} at {}".format(max(bbox_map),
                (bbox_map.index(max(bbox_map)) + 1) * self._period - 1,
            ),
        )
        axis[1].scatter(
            (seg_map.index(max(seg_map)) + 1) * self._period,
            max(seg_map),
            c="g",
            label="best_segm_map50: {:.2f}, at {}".format(max(seg_map),
                (seg_map.index(max(seg_map)) + 1) * self._period - 1,
            ),
        )
        # Update the legend to include the new scatter points/labels                                                                                                            
        axis[1].legend(loc="lower right")
        axis[1].set_title("BBox and Segm mAP over time")
        
        model, dataset = self._output_dir.split("/")[-2:]

        figure.suptitle("InstanceSeg Metrics of {} on {}".format(model, dataset))
        figure.set_figwidth(12)
        figure.set_figheight(8)
        plt.savefig("{}/instance_seg_figure.png".format(self._output_dir))
        plt.close()

        return

    def after_step(self):
        # follow the eval hook period
        next_iter = self.trainer.iter + 1
        if self._period > 0 and next_iter % self._period == 0:
            if next_iter != self.trainer.max_iter:
                self._do_instance_seg_figure()

    def after_train(self):
        if self.trainer.iter + 1 >= self.trainer.max_iter:
            self._do_instance_seg_figure()


class SemSegFigureHook(HookBase):
    def __init__(self, output_dir, eval_period, classes):
        self._output_dir = output_dir
        self._period = eval_period
        self._logger = logging.getLogger("__name__")
        self.classes = classes

    def _do_semseg_figure(self):
        metric_file = "{}/metrics.json".format(self._output_dir)

        # Every line in the metrics.json file is its own json object.
        with open(metric_file, "r") as f:
            lines = [line for line in f.readlines()]

        if len(lines) == 0:
            return

        train_loss = []
        val_loss = []
        acc_dict = {"mean": []}
        iou_dict = {"mean": []}
        for stuff in self.classes:
            acc_dict[stuff] = []
            iou_dict[stuff] = []
        itr_train_loss = []
        itr_val_loss = []
        itr_acc = []
        itr_iou = []
        for line in lines:
            data = json.loads(line)
            itr = data["iteration"]
            if "total_loss" in data:
                train_loss.append(data["total_loss"])
                itr_train_loss.append(itr)
            if "sem_seg/mIoU" in data and "sem_seg/mACC" in data:
                iou_dict["mean"].append(data["sem_seg/mIoU"])
                acc_dict["mean"].append(data["sem_seg/mACC"])
                for stuff in self.classes:
                    iou_dict[stuff].append(data[f"sem_seg/IoU-{stuff}"])
                    acc_dict[stuff].append(data[f"sem_seg/ACC-{stuff}"])
                itr_iou.append(itr)
                itr_acc.append(itr)
            if "validation_loss" in data:
                val_loss.append(data["validation_loss"])
                itr_val_loss.append(itr)

        figure, axis = plt.subplots(1, 3)
        axis[0].plot(itr_train_loss, train_loss, label="loss")
        axis[0].plot(itr_val_loss, val_loss, label="val_loss")
        axis[0].legend(loc="upper right")
        axis[0].set_title("Loss over time")

        line_styles = ["--", "-.", ":"]

        axis[1].plot(itr_iou, iou_dict["mean"], linewidth=3, label="mIoU")
        for idx, stuff in enumerate(self.classes):
            axis[1].plot(
                itr_iou,
                iou_dict[stuff],
                line_styles[idx % len(line_styles)],
                label=f"IoU-{stuff}",
            )
        axis[1].legend(loc="lower right")
        axis[1].set_title("IoU over time")

        axis[2].plot(itr_acc, acc_dict["mean"], linewidth=3, label="mACC")
        for idx, stuff in enumerate(self.classes):
            axis[2].plot(
                itr_acc,
                acc_dict[stuff],
                line_styles[idx % len(line_styles)],
                label=f"ACC-{stuff}",
            )
        axis[2].legend(loc="lower right")
        axis[2].set_title("ACC over time")

        model, dataset = self._output_dir.split("/")[-2:]
        figure.suptitle("SemSeg metrics of {} on {}".format(model, dataset))
        figure.set_figwidth(12)
        figure.set_figheight(8)
        plt.savefig("{}/semseg_figure.png".format(self._output_dir))
        plt.close()
        return

    def after_step(self):
        # follow the eval hook period
        next_iter = self.trainer.iter + 1
        if self._period > 0 and next_iter % self._period == 0:
            if next_iter != self.trainer.max_iter:
                self._do_semseg_figure()

    def after_train(self):
        if self.trainer.iter + 1 >= self.trainer.max_iter:
            self._do_semseg_figure()


# Calculates additional metrics during model validation:
#   * Loss
#   * F1 Score
#   * Recall
#   * Precision
class AdditionalEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader, compute_f1):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
        self._compute_f1 = compute_f1

    def _calculate_metrics(self):
        self._do_loss_eval()
        if self._compute_f1:
            self._do_f1_score_eval()

    def _do_f1_score_eval(self):
        # We set the model to eval mode since in trianing mode the model returns
        # the loss over the input. For F1 score, recall and precision we need
        # the model output for metric calculation.
        training_mode = self._model.training
        self._model.eval()
        metrics_calculator = SegmentationMetrics(activation="0-1", average=True)
        mF1, mPrecision, mRecall = [], [], []
        for inputs in self._data_loader:
            # Validation dataloaders have a batch size of one so it's ok to pull
            # out the first element of the input and prediction lists.
            gt = inputs[0]["sem_seg"].unsqueeze(0)
            pred = (
                self._model(inputs, post_process=False)[0]["sem_seg"]
                .unsqueeze(0)
                .detach()
                .cpu()
            )
            _, dice, precision, recall = metrics_calculator(gt, pred)
            mF1.append(dice)
            mPrecision.append(precision)
            mRecall.append(recall)

        mF1 = np.mean(mF1)
        mPrecision = np.mean(mPrecision)
        mRecall = np.mean(mRecall)
        self.trainer.storage.put_scalar("f1_score", mF1)
        self.trainer.storage.put_scalar("precision", mPrecision)
        self.trainer.storage.put_scalar("recall", mRecall)

        self._model.train(training_mode)

    def _do_loss_eval(self):
        losses = []
        for inputs in self._data_loader:
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar("validation_loss", mean_loss)

        return losses

    def _get_loss(self, data):
        # How loss is calculated on train_loop
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._calculate_metrics()
        self.trainer.storage.put_scalars(timetest=12)

    def after_train(self):
        if self.trainer.iter + 1 >= self.trainer.max_iter:
            self._calculate_metrics()
