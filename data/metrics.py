from typing import Sequence, List, Optional, Mapping, Any, Dict
import numpy as np
from seqio.metrics import Scalar, Text


def cls_accuracy_metric(targets: Sequence[int], predictions):
  predictions = np.array(predictions).astype(np.float32)
  targets = np.array(targets).astype(np.int32)
  score = np.equal(targets, np.argmax(predictions, axis=-1)).astype(np.float32)
  score = np.mean(score)
  return {
    "score": Scalar(score),
  }