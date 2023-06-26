import datetime
import logging
from typing import Any

from tqdm import tqdm
import yaml
import tensorflow as tf

import data_processing
import ltn.wrapper as ltnw
import ltn.utils as ltnu
import pascalpart_theory, pascalpart_theory_log
import pascalpart_domains
import evaluation

logging.getLogger().setLevel(logging.INFO)

with open("config.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

DomGroup = tuple[str]
DomLabel = str

def get_dom_group(constraint: ltnw.Constraint) -> DomGroup:
    return tuple(sorted([dom.label for dom in constraint.doms_feed_dict.values()]))

def get_kwarg_to_dom_label_mapping(constraint: ltnw.Constraint) -> dict[str,str]:
    return {kwarg:dom.label for (kwarg,dom) in constraint.doms_feed_dict.items()}

def get_dom_label_to_kwarg_mapping(constraint: ltnw.Constraint) -> dict[str,str]:
    return {dom.label:kwarg for (kwarg,dom) in constraint.doms_feed_dict.items()}

def apply_key_mapping_to_dict(feed_dict: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    return {mapping[old_key]: values for (old_key,values) in feed_dict.items()}

def train(
        theory: ltnw.Theory,
        test_datasets: evaluation.TestDatasets,
        test_loggers: list[ltnu.logging.MetricsLogger],
        epoch_range: tuple[int,int] = (0,config["epochs"]),
        pretraining: bool = False
):
    constraints = theory.constraints if not pretraining else [cstr for cstr in theory.constraints
                                                             if cstr.label.endswith("groundtruth")]
    print(f"{len(constraints)} constraints:")
    for c in constraints:
        print(c.label)
    for epoch in tqdm(range(*epoch_range), desc="Epochs"):
        theory.constraints[0].operator_config.update_schedule(epoch)
        for _ in tqdm(range(config["training_steps_per_epoch"]), desc="Steps"):
            if pretraining:
                theory.train_step_from_domains([cstr for cstr in theory.constraints 
                        if cstr.label.endswith("groundtruth")])
            else:
                theory.train_step_from_domains()
        theory.reset_metrics()
        evaluation.test_step(theory, test_datasets=test_datasets, loggers=test_loggers, 
                             step=theory.step)

if __name__ == "__main__":
    logging.info("Loading training data.")
    pascal_data = pascalpart_domains.get_pascalpart_data()
    pascal_doms = pascalpart_domains.data_to_domains(pascal_data)

    # loggers
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer_train = tf.summary.create_file_writer(
            "logging/gradient_tape/"+current_time+"/train")
    summary_writer_test = tf.summary.create_file_writer(
            "logging/gradient_tape/"+current_time+"/test")
    tfsummary_logger_train = ltnu.logging.TfSummaryLogger(summary_writer_train)
    tfsummary_logger_test = ltnu.logging.TfSummaryLogger(summary_writer_test)

    df_logger_train = ltnu.logging.DataFrameLogger()
    df_logger_test = ltnu.logging.DataFrameLogger()

    train_loggers = [tfsummary_logger_train, df_logger_train]
    test_loggers = [tfsummary_logger_test, df_logger_test]

    # training
    use_prior_rules = (config["with_mereological_axioms"] or config["with_clustering_axioms"])
    epochs_pretraining = config["epochs_of_pretraining"] if use_prior_rules else 0
    epochs_normal_training = config["epochs"]

    
    logging.info("Building LTN theory.")
    if config["ltn_config"] == "stable_rl" or config["ltn_config"] == "prod_rl":
        theory = pascalpart_theory.get_theory(
                class_to_id=data_processing.get_classes_to_id(),
                part_to_wholes=data_processing.get_part_to_wholes_ontologies(),
                whole_to_parts=data_processing.get_whole_to_parts_ontologies(),
                pascal_doms=pascal_doms,
                metrics_loggers=train_loggers
        )
    else:
        theory = pascalpart_theory_log.get_theory(
                class_to_id=data_processing.get_classes_to_id(),
                part_to_wholes=data_processing.get_part_to_wholes_ontologies(),
                whole_to_parts=data_processing.get_whole_to_parts_ontologies(),
                pascal_doms=pascal_doms,
                metrics_loggers=train_loggers
        )
    logging.info("Loading testing data.")
    test_datasets = evaluation.get_test_datasets()
    

    if epochs_pretraining > 0:
        logging.info("----STARTING PRE-TRAINING----")
        train(theory, test_datasets, test_loggers,
                epoch_range=(0,epochs_pretraining), pretraining=True)

    logging.info("----STARTING TRAINING----")
    train(theory, test_datasets, test_loggers,
            epoch_range=(epochs_pretraining,epochs_normal_training))
    # save csv
    csv_prefix = config["ltn_config"]
    if config["ltn_config"] == "stable_rl":
        csv_prefix += f"_{config['p_universal_quantifier']}"
    df_logger_train.to_csv("logging/dataframes/"+csv_prefix+current_time+"train.csv")
    df_logger_test.to_csv("logging/dataframes/"+csv_prefix+current_time+"test.csv")
