import os
import csv
import collections
import logging
import dataclasses
import tqdm

import numpy as np
import yaml
import h5py

with open("config.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

DATA_FOLDER = "data"

if config["data_category"] == 'vehicle':
    SELECTED_TYPES = np.array(['aeroplane','artifact_wing','body','engine','stern','wheel','bicycle','chain_wheel','handlebar','headlight','saddle','bus','bodywork','door','license_plate','mirror','window','car','motorbike','train','coach','locomotive','boat'])
    CLASSES_FILE = "classes_vehicle.csv"
if config["data_category"] == 'indoor':
    SELECTED_TYPES = np.array(['bottle','body','cap','pottedplant','plant','pot','tvmonitor','screen','chair','sofa','diningtable'])
    CLASSES_FILE = "classes_indoor.csv"
if config["data_category"] == 'animal':
    SELECTED_TYPES = np.array(['person','arm','ear','ebrow','foot','hair','hand','mouth','nose','eye','head','leg','neck','torso','cat','tail','bird','animal_wing','beak','sheep','horn','muzzle','cow','dog','horse','hoof'])
    CLASSES_FILE = "classes_animal.csv"
if config["data_category"] == 'all':
    SELECTED_TYPES = np.array(['handlebar','tvmonitor','ebrow','horn','sofa','mirror','coach','sheep','train','horse','torso','animal_wing','hand','headlight','foot','leg','cow','body','bodywork','bus','hoof','engine','person','bottle','locomotive','tail',
            'chair','screen','wheel','boat','nose','pottedplant','motorbike','arm','bird','pot','cat','diningtable','ear','neck','car','plant','cap','beak','door','artifact_wing','bicycle','license_plate','hair','window','chain_wheel','head','dog','aeroplane',
            'stern','mouth','eye','saddle','muzzle'])
    CLASSES_FILE = "classes.csv"

def get_whole_to_parts_ontologies() -> dict[str,list[str]]:
    """Read whole->parts mereological rules in a dictionary. 
    Only for the selected types of data.

    Returns:
        dict[str,list[str]]: key = whole, values = possible parts that appear in whole.
    """
    with open(os.path.join(DATA_FOLDER,'pascalPartOntology.csv')) as f:
        csv_reader = csv.reader(f)
        whole_to_parts = collections.defaultdict(list)
        for row in csv_reader:
            if row[0] in SELECTED_TYPES:
                whole_to_parts[row[0]] = row[1:]
    return whole_to_parts

def get_part_to_wholes_ontologies() -> dict[str,list[str]]:
    """Read part->wholes mereological rules in a dictionnary.
    Only for the selected types of data.

    Returns:
        dict[str,list[str]]: key = part, values = possible wholes where part appears.
    """
    with open(os.path.join(DATA_FOLDER,'pascalPartOntology.csv')) as f:
        csv_reader = csv.reader(f)
        part_to_wholes = collections.defaultdict(list)
        for row in csv_reader:
            if row[0] in SELECTED_TYPES:
                for t in row[1:]:
                    part_to_wholes[t].append(row[0])
    return part_to_wholes

def get_id_to_classes() -> dict[int,str]:
    """Read id->classes from the disk.

    Returns:
        dict[int,str]: key = class id, value = class label
    """
    with open(os.path.join(DATA_FOLDER,CLASSES_FILE)) as f:
        csv_reader = csv.reader(f)
        id_to_classes = {}
        id_ = 0
        for row in csv_reader:
            id_to_classes[id_] = row[0]
            id_ += 1
    return id_to_classes

def get_classes_to_id() -> dict[str,int]:
    """Read classes->id from the disk.

    Returns:
        dict[str,int]: key = class label, value = class id
    """
    with open(os.path.join(DATA_FOLDER,CLASSES_FILE)) as f:
            csv_reader = csv.reader(f)
            classes_to_id = {}
            id_ = 0
            for row in csv_reader:
                classes_to_id[row[0]] = id_
                id_ += 1
    return classes_to_id

@dataclasses.dataclass
class BoxData:
    """
    Output data of a box in an image.

    Attributes:
        id (int): ID of the box in the dataset.
        type (int): Class id of the object.
        position (np.ndarray): Array of shape [4]. Contains, in order: xmin, ymin, xmax, ymax.
                Optional if `h5_file` is given.
        roi_features (np.ndarray): Output of an object detector.   
                Optional if `h5_file` is given.
        features (np.ndarray): Concatenation of roi_features and position.
    """
    id_: int
    type_: int
    partof_id: int
    pic: int
    _position: np.ndarray = None
    _roi_features: np.ndarray = None
    h5_file: h5py.File = None
    type_str: str = None

    def __post_init__(self):
        self.id_ = int(self.id_)
        self.type_ = int(self.type_)
        self.partof_id = int(self.partof_id)
        self.pic = int(self.pic) 
        if self._roi_features is not None:
            self._roi_features = self._roi_features.astype(np.float32)
        if self._position is not None:
            self._position = self._position.astype(np.int32)

    @property
    def roi_features(self)-> np.ndarray:
        return self.h5_file[str(self.id_)]["roi_features"][()] if self._roi_features is None else self._roi_features

    @property
    def position(self) -> np.ndarray:
        return self.h5_file[str(self.id_)]["position"][()] if self._position is None else self._position
    
    @property 
    def features(self)-> np.ndarray:
        position = (self.position/500).astype(np.float32)
        return np.concatenate([self.roi_features, position], axis=0)

    def __str__(self) -> str:
        return f"Box(id={self.id_},partof={self.partof_id},type={self.type_},type_str={self.type_str})"

@dataclasses.dataclass
class PairedData:
    """_summary_
    """
    id1: int
    id2: int
    ispartof: bool

    def __post_init__(self):
        self.id1 = int(self.id1)
        self.id2 = int(self.id2)
        self.ispartof = bool(self.ispartof)

def get_box_data(
        training: bool = True,
        min_bb_size: int = config["bounding_box_minimal_size"],
        print_type_metrics: bool = False,
        roi_features_in_memory: bool = True
) -> list[BoxData]:
    data_dir = os.path.join(DATA_FOLDER,"trainval") if training else os.path.join(DATA_FOLDER,"test")
    f = h5py.File(os.path.join(data_dir,"box_features.hdf5"))
    all_box_ids = list(f.keys())
    logging.info(f"{len(all_box_ids)} bounding boxes found.")
    class_to_id = get_classes_to_id()
    select_classes = set(class_to_id.keys())
    select_box_data: list[BoxData] = []
    for box_id in tqdm.tqdm(all_box_ids,desc="Reading box ids"):
        type_str = f[box_id].attrs["type"]
        if type_str not in select_classes:
            continue
        type_id = class_to_id[type_str]
        position = f[box_id]["position"][()]
        is_big_enough = lambda coords: np.all((coords[2:4]-coords[:2]) >= min_bb_size)
        if not is_big_enough(position):
            continue
        roi_features = f[box_id]["roi_features"][()] if roi_features_in_memory else None
        select_box_data.append(
                BoxData(id_=box_id, 
                        type_=type_id,
                        pic=f[box_id].attrs["pic"],
                        partof_id=f[box_id].attrs["partof"],
                        _roi_features=roi_features,
                        _position=position, 
                        h5_file=f,
                        type_str=type_str))
    logging.info(f"Using only {len(select_box_data)} bounding boxes (others have types not in the " 
            "experiment category or their sizes are too small).")
    if print_type_metrics:
        log_type_metrics(select_box_data)
    return select_box_data


def filter_box_data_with_labeled_ratio_in_picture_groups(
        box_data: list[BoxData],
        labeled_ratio: float = config["labeled_ratio"], 
        seed: int =config["random_seed"], 
        print_type_metrics: bool = False) -> list[BoxData]:
    pics = set([box.pic for box in box_data])
    n_total = len(pics)
    logging.info(f'Boxes sampled from {n_total} pictures in total.')
    n_keep = int(labeled_ratio*n_total)
    n_remove = n_total - n_keep
    logging.info(f'Removing {n_remove} pictures for semi-supervision.')
    rng = np.random.default_rng(seed)
    select_pics = set(rng.choice(list(pics), n_keep))
    select_box_data = [box for box in box_data if box.pic in select_pics]
    logging.info(f'Filtered data has {len(select_pics)} pictures and {len(select_box_data)} boxes.')
    if print_type_metrics:
        log_type_metrics(select_box_data)
    return select_box_data

def filter_box_data_with_labeled_ratio(
        box_data: list[BoxData],
        labeled_ratio: float = config["labeled_ratio"], 
        seed: int =config["random_seed"]) -> list[BoxData]:
    n_total = len(box_data)
    logging.info(f'Original data has {n_total} boxes.')
    n_keep = int(labeled_ratio*n_total)
    n_remove = n_total - n_keep
    logging.info(f'Removing {n_remove} boxes for semi-supervision.')
    rng = np.random.default_rng(seed)
    select_box_data = rng.choice(box_data, n_keep)
    logging.info(f'Filtered data has {len(select_box_data)} boxes.')
    return select_box_data

def filter_paired_data_with_labeled_ratio(
        paired_data: list[PairedData],
        labeled_ratio: float = config["labeled_ratio"], 
        seed: int =config["random_seed"]) -> list[PairedData]:
    n_total = len(paired_data)
    logging.info(f'Original data has {n_total} pairs.')
    n_keep = int(labeled_ratio*n_total)
    n_remove = n_total - n_keep
    logging.info(f'Removing {n_remove} pairs for semi-supervision.')
    rng = np.random.default_rng(seed)
    select_paired_data = rng.choice(paired_data, n_keep)
    logging.info(f'Filtered data has {len(select_paired_data)} pairs.')
    return select_paired_data

def log_type_metrics(box_data: list[BoxData]) -> None:
    type_id_to_classes = get_id_to_classes()
    types = [box.type_ for box in box_data]
    types_count = collections.Counter(types)
    for type_id, frequency in types_count.items():
        logging.info(f"Type {type_id_to_classes[type_id]}: {frequency} occurrences.")

def get_paired_data(
        box_data: list[BoxData],
        training: bool = True,
        print_partof_metrics: bool = False
    ) -> list[PairedData]:
    data_dir = os.path.join(DATA_FOLDER,"trainval") if training else os.path.join(DATA_FOLDER,"test")
    f = h5py.File(os.path.join(data_dir,"pairs_partof.hdf5"))
    select_ids = set([box.id_ for box in box_data])
    paired_data: list[PairedData] = []
    for box1_id in tqdm.tqdm(select_ids,desc="Processing pairs"):
        box1_node = f[str(box1_id)]
        for box2_id in box1_node.keys():
            if int(box2_id) not in select_ids:
                continue
            paired_data.append(PairedData(box1_id, box2_id, box1_node[box2_id].attrs["ispartof"]))
    logging.info(f"Using {len(paired_data)} pairs of bounding boxes.")
    if print_partof_metrics:
        id_to_box = {box.id_:box for box in box_data}
        log_partof_metrics(id_to_box, paired_data)
    return paired_data

def log_partof_metrics(id_to_box: dict[str,BoxData], paired_data: list[PairedData]) -> None:
    pairs_ispartof = [pair for pair in paired_data if pair.ispartof]
    pairs_notpartof = [pair for pair in paired_data if not pair.ispartof]
    cont_ratios_ispartof = [containment_ratio_of_bb1_in_bb2(
            id_to_box[pair.id1].position, id_to_box[pair.id2].position, batch_mode=False)
            for pair in pairs_ispartof]
    cont_ratios_notpartof = [containment_ratio_of_bb1_in_bb2(
            id_to_box[pair.id1].position, id_to_box[pair.id2].position, batch_mode=False)
            for pair in pairs_notpartof]
    logging.info(f'{len(pairs_ispartof)} positive examples '
            f'(avg. containment ratio = {np.mean(cont_ratios_ispartof):.2f}) and '
            f'{len(pairs_notpartof)} negative examples '
            f'(avg. containment ratio = {np.mean(cont_ratios_notpartof):.2f}) for ' 
            'the partOf relationship.')


def containment_ratio_of_bb1_in_bb2(bb1: np.ndarray, bb2: np.ndarray, batch_mode: bool) -> float:
    """Containment ratio of bb1 in bb2.
    
    Args:
        bb1 (np.ndarray): features of bb1 (`BoxData.features`)
        bb2 (np.ndarray): features of bb2 (`BoxData.features`)

    Returns:
        float: area_intersection(bb1,bb2)/area(bb1)
    """
    if batch_mode:
        bb1_area = (bb1[:,-2] - bb1[:,-4]) * (bb1[:,-1] - bb1[:,-3])
        w_intersec = np.amin([bb1[:,-2], bb2[:,-2]], axis=0) - np.amax([bb1[:,-4], bb2[:,-4]], axis=0)
        w_intersec = np.where(w_intersec>0, w_intersec, 0)
        h_intersec = np.amin([bb1[:,-1], bb2[:,-1]], axis=0) - np.amax([bb1[:,-3], bb2[:,-3]], axis=0)
        h_intersec = np.where(h_intersec>0, h_intersec, 0)
        bb_area_intersection = w_intersec * h_intersec
        return bb_area_intersection/bb1_area
    else:
        bb1_area = (bb1[-2] - bb1[-4]) * (bb1[-1] - bb1[-3])
        w_intersec = np.amax([0, np.amin([bb1[-2], bb2[-2]]) - np.amax([bb1[-4], bb2[-4]])])
        h_intersec = np.amax([0, np.amin([bb1[-1], bb2[-1]]) - np.amax([bb1[-3], bb2[-3]])])
        bb_area_intersection = w_intersec * h_intersec
        return bb_area_intersection/bb1_area