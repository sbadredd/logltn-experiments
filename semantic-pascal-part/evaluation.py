from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Callable, Union
import logging
from tqdm import tqdm
import yaml
import sklearn.metrics
import tensorflow as tf
import numpy as np

import ltn.wrapper as ltnw
import ltn.utils as ltnu
import data_processing

with open("config.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

@dataclass
class TestData:
    id_to_box: dict[str, data_processing.BoxData]
    pairs_all: list[data_processing.PairedData]
    type_id_to_str: dict[int, str]
    type_str_to_id: dict[str, int]
    
    def id_to_features(self, id_: int) -> np.ndarray:
        return self.id_to_box[int(id_)].features

@dataclass
class TestDatasets:
    """Here we don't premap the features"""
    ds_x_and_types: tf.data.Dataset
    ds_x1x2_and_ispartof: tf.data.Dataset
    map_fn_id_to_features: Callable
    data: TestData

def get_test_datasets() -> TestDatasets:
    logging.info(f"----TEST DATA----")
    box_all = data_processing.get_box_data(training=False, print_type_metrics=True)
    id_to_box = {box.id_:box for box in box_all}
    pairs_all = data_processing.get_paired_data(box_all, training=False, print_partof_metrics=True)
    
    data = TestData(id_to_box=id_to_box, pairs_all=pairs_all, 
            type_id_to_str=data_processing.get_id_to_classes(), 
            type_str_to_id=data_processing.get_classes_to_id())
    id_to_features_fn = lambda id_: tf.numpy_function(data.id_to_features, 
            inp=[id_],Tout=tf.float32)

    x = [box.id_ for box in box_all]
    types = [box.type_ for box in box_all]
    ds_x_and_types = tf.data.Dataset.from_tensor_slices((x,types))\
            .batch(config["test_minibatch_size"])
    
    x1 = [pair.id1 for pair in pairs_all]
    x2 = [pair.id2 for pair in pairs_all]
    ispartof = [pair.ispartof for pair in pairs_all]
    ds_x1x2_and_ispartof = tf.data.Dataset.from_tensor_slices((x1,x2,ispartof))\
            .batch(config["test_minibatch_size"])
    return TestDatasets(ds_x_and_types=ds_x_and_types, ds_x1x2_and_ispartof=ds_x1x2_and_ispartof,
            map_fn_id_to_features=id_to_features_fn, data=data)

def count_reasonable_false_partof_positives(
        false_positives_per_types: dict[tuple[str,str], int],
        part_to_wholes: dict[str, list[str]] = data_processing.get_part_to_wholes_ontologies(),
        reverse: bool = False
) -> int:
    total = 0
    for types, count in false_positives_per_types.items():
        if not reverse and types[1] in part_to_wholes[types[0]]:
            total += count
        elif reverse and types[1] not in part_to_wholes[types[0]]:
            total += count
    return total

def evaluate_partof_and_reasoning(
        theory: ltnw.Theory,
        datasets: TestDatasets
) -> PartOfMetrics:
    all_y_score, all_y_true = [], []
    classification_threshold = 0.5
    types_count_false_neg = defaultdict(lambda: 0)
    types_count_false_pos = defaultdict(lambda: 0)
    for x1, x2, y_true in tqdm(datasets.ds_x1x2_and_ispartof, desc="Computing PartOf(x1,x2) for test data"):
        x1_features = tf.stack([datasets.map_fn_id_to_features(x1i) for x1i in x1])
        x2_features = tf.stack([datasets.map_fn_id_to_features(x2i) for x2i in x2])
        y_score = tf.squeeze(theory.grounding.predicates["partOf"].model([x1_features,x2_features]))
        all_y_score.append(y_score)
        all_y_true.append(y_true)
        # analyze types mispredicted
        y_pred = y_score > classification_threshold
        false_pos = tf.math.logical_and(y_pred, tf.logical_not(y_true))
        false_neg = tf.math.logical_and(tf.logical_not(y_pred), y_true)
        for (x1i, x2i) in zip(x1[false_neg].numpy(), x2[false_neg].numpy()):
            box1, box2 = datasets.data.id_to_box[x1i], datasets.data.id_to_box[x2i]
            types_count_false_neg[(box1.type_str, box2.type_str)] += 1
        for (x1i, x2i) in zip(x1[false_pos].numpy(), x2[false_pos].numpy()):
            box1, box2 = datasets.data.id_to_box[x1i], datasets.data.id_to_box[x2i]
            types_count_false_pos[(box1.type_str, box2.type_str)] += 1
    y_score, y_true = tf.concat(all_y_score, axis=0), tf.concat(all_y_true, axis=0)
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, y_score)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    pr_auc = sklearn.metrics.average_precision_score(y_true, y_score)
    false_neg = sum(types_count_false_neg.values())
    reasonable_false_pos = count_reasonable_false_partof_positives(types_count_false_pos)
    unreasonable_false_pos = count_reasonable_false_partof_positives(types_count_false_pos, reverse=True)
    return PartOfMetrics(roc_auc, pr_auc, false_neg, reasonable_false_pos, unreasonable_false_pos)

def test_step(
        theory: ltnw.Theory,
        test_datasets: TestDatasets,
        loggers: list[ltnu.MetricsLogger],
        step: int = None
) -> None:
    partof_metrics = evaluate_partof_and_reasoning(theory, test_datasets)
    type_metrics = evaluate_type(theory, test_datasets)
    partof_metrics.print()
    type_metrics.print()
    dict_metrics = {**partof_metrics.as_dict(), **type_metrics.as_dict()} 
    for logger in loggers:
        logger.log_dict_of_values(dict_metrics, step=step)

@dataclass
class PartOfMetrics:
    roc_auc: float
    pr_auc: float
    false_neg: int
    reasonable_false_pos: int
    unreasonable_false_pos: int

    def print(self):
        print("PartOf ROC AUC: %.4f - PartOf PR AUC: %.4f" % (self.roc_auc, self.pr_auc))
        print("False Negatives: %i - Reasonable False Positives: %i"
                " - Unreasonable False Positives %i"
                    %(self.false_neg, self.reasonable_false_pos, self.unreasonable_false_pos))

    def as_dict(self):
        return {"partof_roc_auc": self.roc_auc, "partof_pr_auc":self.pr_auc, 
                "false_neg": self.false_neg, "reasonable_false_pos": self.reasonable_false_pos,
                "unreasonable_false_pos": self.unreasonable_false_pos}

def evaluate_partof(
        theory: ltnw.Theory,
        datasets: TestDatasets
) -> PartOfMetrics:
    all_y_score, all_y_true = [], []
    for x1, x2, y_true in tqdm(datasets.ds_x1x2_and_ispartof, desc="Computing PartOf(x1,x2) for test data"):
        x1_features = tf.stack([datasets.map_fn_id_to_features(x1i) for x1i in x1])
        x2_features = tf.stack([datasets.map_fn_id_to_features(x2i) for x2i in x2])
        y_score = tf.squeeze(theory.grounding.predicates["partOf"].model([x1_features,x2_features]))
        all_y_score.append(y_score)
        all_y_true.append(y_true)
    y_score, y_true = tf.concat(all_y_score, axis=0), tf.concat(all_y_true, axis=0)
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, y_score)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    pr_auc = sklearn.metrics.average_precision_score(y_true, y_score)
    return PartOfMetrics(roc_auc, pr_auc)

@dataclass
class TypeMetrics:
    accuracy: float
    balanced_accuracy: float

    def print(self):
        print("Type acc: %.4f - Type balanced acc: %.4f" \
                % (self.accuracy, self.balanced_accuracy))
    
    def as_dict(self):
        return {"type_acc": self.accuracy, "type_balanced_acc":self.balanced_accuracy}

def evaluate_type(
        theory: ltnw.Theory,
        datasets: TestDatasets
) -> TypeMetrics:
    all_y_pred, all_y_true = [], []
    for x, y_true in tqdm(datasets.ds_x_and_types, desc="Computing Type(x) for test data"):
        x_features = tf.stack([datasets.map_fn_id_to_features(xi) for xi in x])
        logits = theory.grounding.predicates["is_type"].logits_model([x_features])
        all_y_pred.append(tf.argmax(logits, axis=-1))
        all_y_true.append(y_true)
    y_pred = tf.concat(all_y_pred,axis=0)
    y_true = tf.concat(all_y_true, axis=0)
    acc = sklearn.metrics.accuracy_score(y_true, y_pred)
    balanced_acc = sklearn.metrics.balanced_accuracy_score(y_true, y_pred)
    return TypeMetrics(acc, balanced_acc)
