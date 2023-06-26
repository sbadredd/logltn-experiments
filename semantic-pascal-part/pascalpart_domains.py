import yaml
from dataclasses import dataclass
import logging
import copy

import numpy as np
import tensorflow as tf
import ltn.wrapper as ltnw

import data_processing

with open("config.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

@dataclass
class PascalPartDomains:
    x_lbl: ltnw.Domain
    types_lbl: ltnw.Domain
    x1_ispart: ltnw.Domain
    x2_ispart: ltnw.Domain
    x1_isnotpart: ltnw.Domain
    x2_isnotpart: ltnw.Domain
    x_all: ltnw.Domain
    x1_all: ltnw.Domain
    x2_all: ltnw.Domain

    def as_list(self) -> list[ltnw.Domain]:
        return [self.x_lbl, self.types_lbl, self.x1_ispart, self.x2_ispart, self.x1_isnotpart, 
                self.x2_isnotpart, self.x_all, self.x1_all, self.x2_all]

@dataclass
class PascalPartData:
    id_to_box: dict[int,data_processing.BoxData]
    ids_labeled: list[int]
    pairs_all: list[data_processing.PairedData]
    pairs_labeled: list[data_processing.PairedData]

    @property
    def ids_all(self) -> list[int]:
        return list(self.id_to_box.keys())

    @property
    def types_labeled(self) -> list[int]:
        return [self.id_to_box[id_].type_ for id_ in self.ids_labeled]

    @property
    def x1x2_all(self) -> list[tuple[int,int]]:
        return [(pair.id1,pair.id2) for pair in self.pairs_all]
    
    @property
    def x1x2_ispart(self) -> list[tuple[int,int]]:
        return [(pair.id1,pair.id2) for pair in self.pairs_labeled if pair.ispartof]

    @property
    def x1x2_isnotpart(self) -> list[tuple[int,int]]:
        return [(pair.id1,pair.id2) for pair in self.pairs_labeled if not pair.ispartof]

    def id_to_features(self, id_: int) -> np.ndarray:
        return self.id_to_box[int(id_)].features

def get_pascalpart_data(
        labeled_ratio: float = config["labeled_ratio"]
) -> PascalPartData:
    logging.info(f"----ALL DATA----") # all data used for mereological axioms
    box_all = data_processing.get_box_data(training=True)
    id_to_box = {box.id_:box for box in box_all}
    pairs_all = data_processing.get_paired_data(box_all, training=True)
    logging.info(f"----LABELED DATA----") # subset used as groundtruth examples for partof and type
    box_labeled = data_processing.filter_box_data_with_labeled_ratio(box_all, 
            labeled_ratio=labeled_ratio)
    data_processing.log_type_metrics(box_labeled)
    ids_labeled = [box.id_ for box in box_labeled]
    pairs_labeled = data_processing.filter_paired_data_with_labeled_ratio(pairs_all)
    data_processing.log_partof_metrics(id_to_box, pairs_labeled)    
    return PascalPartData(id_to_box=id_to_box, ids_labeled=ids_labeled, pairs_all=pairs_all,
            pairs_labeled=pairs_labeled)

def data_to_domains(
        data: PascalPartData,
        minibatch_size: int = config["minibatch_size"],
        shuffle: bool = True, 
        shuffle_buffer_size: int = config["shuffle_buffer_size"],
        with_mapping: bool = True
        ) -> PascalPartDomains:
    ds_params = ltnw.DatasetParams(shuffle=shuffle, shuffle_buffer_size=shuffle_buffer_size, 
            minibatch_size=minibatch_size)
    ds_params_with_map = copy.copy(ds_params)
    if with_mapping:
        ds_params_with_map.map_fn = lambda id_: tf.numpy_function(data.id_to_features, 
                inp=[id_],Tout=tf.float32)
    x_lbl = ltnw.Domain("x_lbl", data.ids_labeled, dataset_params=ds_params_with_map)
    types_lbl = ltnw.Domain("types_lbl", data.types_labeled, dataset_params=ds_params)
    ltnw.diag_domains([x_lbl, types_lbl])
    x1_ispart = ltnw.Domain("x1_ispart", [x1x2[0] for x1x2 in data.x1x2_ispart], dataset_params=ds_params_with_map)
    x2_ispart = ltnw.Domain("x2_ispart", [x1x2[1] for x1x2 in data.x1x2_ispart], dataset_params=ds_params_with_map)
    ltnw.diag_domains([x1_ispart,x2_ispart])
    x1_isnotpart = ltnw.Domain("x1_isnotpart", [x1x2[0] for x1x2 in data.x1x2_isnotpart], dataset_params=ds_params_with_map)
    x2_isnotpart = ltnw.Domain("x2_isnotpart", [x1x2[1] for x1x2 in data.x1x2_isnotpart], dataset_params=ds_params_with_map)
    ltnw.diag_domains([x1_isnotpart,x2_isnotpart])
    ds_params_with_map_and_drop = copy.copy(ds_params)
    if with_mapping:
        ds_params_with_map_and_drop.map_fn = lambda id_: tf.numpy_function(data.id_to_features, 
                inp=[id_],Tout=tf.float32)
    ds_params_with_map_and_drop.drop_remainder = True 
                    # -> `drop_remainder=True` is makes the batches easier to handle 
                    # by the curriculum learning functions. It is ok to drop remainders given that
                    # there are many examples.
    x_all = ltnw.Domain("x_all", data.ids_all, dataset_params=ds_params_with_map_and_drop)
    x1_all = ltnw.Domain("x1_all", [x1x2[0] for x1x2 in data.x1x2_all], 
            dataset_params=ds_params_with_map_and_drop)
    x2_all = ltnw.Domain("x2_all", [x1x2[1] for x1x2 in data.x1x2_all], 
            dataset_params=ds_params_with_map_and_drop)
    ltnw.diag_domains([x1_all,x2_all])
    return PascalPartDomains(x_lbl=x_lbl, types_lbl=types_lbl, x1_ispart=x1_ispart, 
            x2_ispart=x2_ispart, x1_isnotpart=x1_isnotpart, x2_isnotpart=x2_isnotpart, 
            x_all=x_all, x1_all=x1_all, x2_all=x2_all)

