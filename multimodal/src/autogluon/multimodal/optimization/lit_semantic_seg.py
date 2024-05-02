import logging
from typing import Callable, Dict

import torch
import torchmetrics
from transformers.models.mask2former.modeling_mask2former import Mask2FormerLoss

from ..constants import CLASS_LOGITS, LOGITS, SEMANTIC_MASK, WEIGHT
from ..models.utils import run_model
from .lit_module import LitModule
from .semantic_seg_metrics import Multiclass_IoU

logger = logging.getLogger(__name__)


class SemanticSegmentationLitModule(LitModule):
    """
    Control the loops for training, evaluation, and prediction. This module is independent of
    the model definition. This class inherits from the Pytorch Lightning's LightningModule:
def _compute_loss(self, y_true, y_pred):
    loss = self.loss_fn(y_true, y_pred)
    return loss
            if isinstance(self.loss_func, Mask2FormerLoss):
                mask_labels = [mask_labels.to(per_output[LOGITS]) for mask_labels in kwargs["mask_labels"]]
                dict_loss = self.loss_func(
                    masks_queries_logits=per_output[LOGITS],  # bs, num_mask_tokens, height, width
                    class_queries_logits=per_output[CLASS_LOGITS],  # bs, num_mask_tokens, num_classes
                    mask_labels=mask_labels,
                    class_labels=kwargs["class_labels"],
                )
                for v in dict_loss.values():
                    loss += v
            else:
                loss += (
                    self.loss_func(
                        input=per_output[LOGITS],
                        target=label,
                    )
                    * weight
                )
        return loss

    def _compute_metric_score(
        self,
        metric: torchmetrics.Metric,
        custom_metric_func: Callable,
        logits: torch.Tensor,
        label: torch.Tensor,
        **kwargs,
    ):
        if isinstance(metric, Multiclass_IoU):
            metric.update(kwargs["semantic_masks"], label)
        else:
            metric.update(logits.float(), label)

    def _shared_step(
        self,
        batch: Dict,
    ):
        label = batch[self.model.label_key]
        # prepare_targets
        output = run_model(self.model, batch)
        if isinstance(self.loss_func, Mask2FormerLoss):
            loss = self._compute_loss(
                output=output,
                label=label,
                mask_labels=batch[self.model.mask_label_key],
                class_labels=batch[self.model.class_label_key],
            )
        else:
            loss = self._compute_loss(
                output=output,
                label=label,
            )

        return output, loss

    def validation_step(self, batch, batch_idx, **kwargs):
        """
class CustomModel(tf.keras.Model):
    def validation_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        loss = self.compiled_loss(y, y_pred)
        self.compiled_metrics.update_state(y, y_pred)
        return {"val_loss": loss}
    # Add a missing closing brace here