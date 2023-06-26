import yaml
from dataclasses import dataclass, field
from typing import Callable

import ltn
import ltn.wrapper as ltnw
import ltn.utils as ltnu
import tensorflow as tf
import numpy as np

import data_processing
from pascalpart_domains import PascalPartDomains


with open("config.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

def get_theory(
    class_to_id: dict[str,int],
    part_to_wholes: dict[str,list[str]], 
    whole_to_parts: dict[str, list[str]],
    pascal_doms: PascalPartDomains,
    metrics_loggers: list[ltnu.logging.MetricsLogger] = None,
    log_every_n_step: int = config["training_steps_per_epoch"],
    op_config: str = config["ltn_config"]
):
    if op_config == "log_ltn":
        operator_config = get_log_operator_config()
    elif op_config == "log_ltn_sum":
        operator_config = get_logsum_operator_config()
    elif op_config == "log_ltn_lse":
        operator_config = get_loglse_operator_config()
    elif op_config == "log_ltn_max":
        operator_config = get_logmax_operator_config()
    grounding = get_grounding(class_to_id)
    constraints = get_constraints(part_to_wholes, whole_to_parts, pascal_doms, grounding, operator_config)

    theory = ltnw.Theory(
            constraints=constraints,
            grounding=grounding,
            formula_aggregator=operator_config.and_aggreg,
            metrics_loggers=metrics_loggers,
            log_every_n_step=log_every_n_step
    )
    return theory

def get_logsum_operator_config() -> ltnw.OperatorConfig:
    alpha = config["a_existential_quantifier"]
    not_ = None
    and_ = ltn.log.Wrapper_Connective(ltn.log.fuzzy_ops.And_Sum())
    or_ = ltn.log.Wrapper_Connective(ltn.log.fuzzy_ops.Or_LogMeanExp(alpha=alpha))
    implies = None
    forall = ltn.log.Wrapper_Quantifier(ltn.log.fuzzy_ops.Aggreg_Sum(),semantics="forall")
    and_aggreg = ltn.log.Wrapper_Formula_Aggregator(ltn.log.fuzzy_ops.Aggreg_Sum())
    exists = ltn.log.Wrapper_Quantifier(ltn.log.fuzzy_ops.Aggreg_LogMeanExp(alpha=alpha), semantics="exists")
    or_aggreg = ltn.log.Wrapper_Formula_Aggregator(ltn.log.fuzzy_ops.Aggreg_LogMeanExp(alpha=alpha))
    op_config = ltnw.OperatorConfig(not_=not_, and_=and_, or_=or_, implies=implies, 
        exists=exists, forall=forall, and_aggreg=and_aggreg, or_aggreg=or_aggreg)
    if not config["schedule_p_existential_quantifier"] == "None":    
        set_exist_schedule(op_config)
    return op_config

def get_loglse_operator_config() -> ltnw.OperatorConfig:
    alpha = config["a_existential_quantifier"]
    not_ = None
    and_ = ltn.log.Wrapper_Connective(ltn.log.fuzzy_ops.And_Sum())
    or_ = ltn.log.Wrapper_Connective(ltn.log.fuzzy_ops.Or_LogSumExp(alpha=alpha))
    implies = None
    forall = ltn.log.Wrapper_Quantifier(ltn.log.fuzzy_ops.Aggreg_Mean(),semantics="forall")
    and_aggreg = ltn.log.Wrapper_Formula_Aggregator(ltn.log.fuzzy_ops.Aggreg_Mean())
    exists = ltn.log.Wrapper_Quantifier(ltn.log.fuzzy_ops.Aggreg_LogSumExp(alpha=alpha), semantics="exists")
    or_aggreg = ltn.log.Wrapper_Formula_Aggregator(ltn.log.fuzzy_ops.Aggreg_LogSumExp(alpha=alpha))
    op_config = ltnw.OperatorConfig(not_=not_, and_=and_, or_=or_, implies=implies, 
        exists=exists, forall=forall, and_aggreg=and_aggreg, or_aggreg=or_aggreg)
    if not config["schedule_p_existential_quantifier"] == "None":    
        set_exist_schedule(op_config)
    return op_config

def get_logmax_operator_config() -> ltnw.OperatorConfig:
    not_ = None
    and_ = ltn.log.Wrapper_Connective(ltn.log.fuzzy_ops.And_Sum())
    or_ = ltn.log.Wrapper_Connective(ltn.log.fuzzy_ops.Or_Max())
    implies = None
    forall = ltn.log.Wrapper_Quantifier(ltn.log.fuzzy_ops.Aggreg_Mean(),semantics="forall")
    and_aggreg = ltn.log.Wrapper_Formula_Aggregator(ltn.log.fuzzy_ops.Aggreg_Mean())
    exists = ltn.log.Wrapper_Quantifier(ltn.log.fuzzy_ops.Aggreg_Max(), semantics="exists")
    or_aggreg = ltn.log.Wrapper_Formula_Aggregator(ltn.log.fuzzy_ops.Aggreg_Max())
    op_config = ltnw.OperatorConfig(not_=not_, and_=and_, or_=or_, implies=implies, 
        exists=exists, forall=forall, and_aggreg=and_aggreg, or_aggreg=or_aggreg)
    return op_config

def get_log_operator_config() -> ltnw.OperatorConfig:
    alpha = config["a_existential_quantifier"]
    not_ = None
    and_ = ltn.log.Wrapper_Connective(ltn.log.fuzzy_ops.And_Sum())
    or_ = ltn.log.Wrapper_Connective(ltn.log.fuzzy_ops.Or_LogMeanExp(alpha=alpha))
    implies = None
    forall = ltn.log.Wrapper_Quantifier(ltn.log.fuzzy_ops.Aggreg_Mean(),semantics="forall")
    and_aggreg = ltn.log.Wrapper_Formula_Aggregator(ltn.log.fuzzy_ops.Aggreg_Mean())
    exists = ltn.log.Wrapper_Quantifier(ltn.log.fuzzy_ops.Aggreg_LogMeanExp(alpha=alpha), semantics="exists")
    or_aggreg = ltn.log.Wrapper_Formula_Aggregator(ltn.log.fuzzy_ops.Aggreg_LogMeanExp(alpha=alpha))
    op_config = ltnw.OperatorConfig(not_=not_, and_=and_, or_=or_, implies=implies, 
        exists=exists, forall=forall, and_aggreg=and_aggreg, or_aggreg=or_aggreg)
    if not config["schedule_p_existential_quantifier"] == "None":    
        set_exist_schedule(op_config)
    return op_config

def get_grounding(class_to_id: dict[str,int]) -> ltnw.Grounding:
    # Predicates for partOf(x,y) and is_type(x,type)
    type_model = TypeModel(n_classes=len(class_to_id))
    partof_model = PartOfModel(type_model)
    partOf = ltn.log.Predicate.FromLogits(partof_model, 
            activation_function="sigmoid", with_class_indexing=False)
    is_type = ltn.log.Predicate.FromLogits(type_model,
             activation_function="softmax", with_class_indexing=True)
    predicates = {"partOf": partOf, "is_type": is_type}
    # Constants for indexing each type
    constants = {class_: ltn.Constant(id_, trainable=False) for class_, id_ in class_to_id.items()}
    # Variable over the classes, the possible types
    variables = {"classes": ltn.Variable.from_constants("classes", list(constants.values()))}
    return ltnw.Grounding(predicates=predicates, constants=constants, variables=variables)

def get_constraints(
        part_to_wholes: dict[str,list[str]], 
        whole_to_parts: dict[str, list[str]],
        pascal_doms: PascalPartDomains,
        grounding: ltnw.Grounding,
        operator_config: ltnw.OperatorConfig
        ) -> list[ltnw.Constraint]:
    constraints = []
    constraints += get_groundtruth_constraints(pascal_doms, grounding, operator_config)
    if config["with_mereological_axioms"]:
        constraints += get_mereological_constraints(part_to_wholes, whole_to_parts, pascal_doms, grounding,
                operator_config)
    if config["with_clustering_axioms"]:
        constraints += get_clustering_constraints(pascal_doms, grounding, operator_config)
    return constraints

def get_groundtruth_constraints(
        pascal_doms: PascalPartDomains,
        grounding: ltnw.Grounding,
        operator_config: ltnw.OperatorConfig
):
    constraints = []
    constraints.append(Rule_TypeGroundtruth("type_groundtruth", grounding, operator_config, 
            doms_feed_dict={"x": pascal_doms.x_lbl, "types": pascal_doms.types_lbl}))
    constraints.append(Rule_IsPartOfGroundtruth("is_partof_groundtruth", grounding, operator_config, 
            doms_feed_dict={"x1": pascal_doms.x1_ispart, "x2": pascal_doms.x2_ispart}))
    constraints.append(Rule_NotPartOfGroundtruth("not_partof_groundtruth", grounding, operator_config, 
            doms_feed_dict={"x1": pascal_doms.x1_isnotpart, "x2": pascal_doms.x2_isnotpart}))
    return constraints

def get_mereological_constraints(
        part_to_wholes: dict[str,list[str]], 
        whole_to_parts: dict[str, list[str]],
        pascal_doms: PascalPartDomains,
        grounding: ltnw.Grounding,
        operator_config: ltnw.OperatorConfig
):
    constraints = []
    constraints.append(Rule_PartOfAntiSymmetry("partof_antisymmetry", grounding, operator_config,
            doms_feed_dict={"x1": pascal_doms.x1_all, "x2": pascal_doms.x2_all}))
    constraints.append(Rule_PartOfAntiReflexive("partof_antireflexive", grounding, operator_config,
            doms_feed_dict={"x": pascal_doms.x_all}))
    for whole, parts in whole_to_parts.items():
        if not parts: # no part in the whole
            continue
        constraints.append(Rule_WholeToParts(f"whole_to_parts_{whole}", grounding, operator_config,
                doms_feed_dict={"x1": pascal_doms.x1_all, "x2": pascal_doms.x2_all},
                whole=whole, parts=parts))
    for part, wholes in part_to_wholes.items():
        constraints.append(Rule_PartToWholes(f"part_to_wholes_{part}", grounding, operator_config,
                doms_feed_dict={"x1": pascal_doms.x1_all, "x2": pascal_doms.x2_all},
                part=part, wholes=wholes))
    return constraints

def get_clustering_constraints(
        pascal_doms: PascalPartDomains,
        grounding: ltnw.Grounding,
        operator_config: ltnw.OperatorConfig
):
    constraints = []
    constraints.append(Rule_OneClassPerSample("one_class_per_sample", grounding, operator_config, 
            doms_feed_dict={"x": pascal_doms.x_all}))
    return constraints

class Rule_TypeGroundtruth(ltnw.Constraint):
    """
    Axiom for the bounding boxes x and their types that verify is_type(x,types).
    `x` and `types` are expected to be diagged.
    """
    def formula(self, x: ltn.Variable, types: ltn.Variable, 
            **kwargs) -> ltn.Formula:
        """
        Args:
            x (ltn.Variable): features of bounding boxes. Expected to be diagged with `types`.
            types (ltn.Variable): their respective type id. Expected to be diagged with `x`.

        Returns:
            ltn.Formula: resulting ltn formula
        """        
        P = self.grounding.predicates
        op = self.operator_config
        res = P["is_type"].log([x,types])
        res = op.forall([x,types], res)
        return res

    def open_formula(self, *args, **kwargs) -> ltn.Formula:
        return self.formula(*args, keep_formula_open=True, **kwargs)


class Rule_IsPartOfGroundtruth(ltnw.Constraint):
    """
    Axiom for the pairs x1, x2 of bounding boxes that verify partOf(x1,x2).
    `x1` and `x2` are expected to be diagged. They are positive examples of the data.
    """

    def formula(self, x1: ltn.Variable, x2: ltn.Variable, 
            **kwargs) -> ltn.Formula:
        """
        Args:
            x1 (ltn.Variable): features of x1. Expected to be diagged with `x2`.
            x2 (ltn.Variable): features of x2. Expected to be diagged with `x1`.

        Returns:
            ltn.Formula: resulting ltn formula
        """
        P = self.grounding.predicates
        op = self.operator_config
        res = P["partOf"].log([x1, x2])
        res = op.forall([x1, x2], res)
        return res

    def open_formula(self, *args, **kwargs) -> ltn.Formula:
        return self.formula(*args, keep_formula_open=True, **kwargs)


class Rule_NotPartOfGroundtruth(ltnw.Constraint):
    """
    Axiom for the pairs x1, x2 of bounding boxes that do not verify partOf(x1,x2)
    `x1` and `x2` are expected to be diagged. They are negative examples of the data.
    """
    def formula(self, x1: ltn.Variable, x2: ltn.Variable, keep_formula_open: bool = False,
            **kwargs) -> ltn.Formula:
        """
        Args:
            x1 (ltn.Variable): features of x1. Expected to be diagged with `x2`.
            x2 (ltn.Variable): features of x2. Expected to be diagged with `x1`.

        Returns:
            ltn.Formula: resulting ltn formula
        """
        P = self.grounding.predicates
        op = self.operator_config
        res = P["partOf"].nlog([x1, x2])
        res = op.forall([x1, x2], res)
        return res

    def open_formula(self, *args, **kwargs) -> ltn.Formula:
        return self.formula(*args, keep_formula_open=True, **kwargs)

class Rule_PartOfAntiSymmetry(ltnw.Constraint):
    """
    Axiom for saying that partOf(x1,x2) -> Not partof(x2,x1).
    Re-formulated for log-ltn as Not(partof(x1,x2)) OR Not(partof(x2,x1))
    `x1` and `x2` are expected to be diagged. They are bounding boxes coming from
    the same pictures.
    """
    def formula(self, x1: ltn.Variable, x2: ltn.Variable, 
            **kwargs) -> ltn.Formula:
        """
        Args:
            x1 (ltn.Variable): features of x1. Expected to be diagged with `x2`.
            x2 (ltn.Variable): features of x2. Expected to be diagged with `x1`.

        Returns:
            ltn.Formula: resulting ltn formula
        """
        P = self.grounding.predicates
        op = self.operator_config
        res = op.or_(P["partOf"].nlog([x1, x2]), P["partOf"].nlog([x2, x1]))
        res = op.forall([x1, x2], res)
        return res

    def open_formula(self, *args, **kwargs) -> ltn.Formula:
        return self.formula(*args, keep_formula_open=True, **kwargs)

class Rule_PartOfAntiReflexive(ltnw.Constraint):
    """
    Axiom for saying that partOf(x,x) is not possible.
    """
    def formula(self, x: ltn.Variable, **kwargs) -> ltn.Formula:
        """
        Args:
            x (ltn.Variable): features of a bounding box

        Returns:
            ltn.Formula: resulting ltn formula
        """
        P = self.grounding.predicates
        op = self.operator_config
        res = P["partOf"].nlog([x,x])
        res = op.forall(x, res)
        return res

    def open_formula(self, *args, **kwargs) -> ltn.Formula:
        return self.formula(*args, keep_formula_open=True, **kwargs)
    
class Rule_OneClassPerSample(ltnw.Constraint):
    def formula(self, x: ltn.Variable, **kwargs) -> ltn.Formula:
        P = self.grounding.predicates
        var_classes = self.grounding.variables["classes"]
        op = self.operator_config
        res = op.exists(var_classes, P["is_type"].log([x, var_classes]))
        res = op.forall(x, res)
        return res

    def open_formula(self, *args, **kwargs):
        return self.formula(*args, keep_formula_open=True, **kwargs)

class Rule_OneSamplePerClass(ltnw.Constraint):
    def formula(self, x: ltn.Variable, **kwarga) -> ltn.Formula:
        P = self.grounding.predicates
        var_classes = self.grounding.variables["classes"]
        op = self.operator_config
        res = op.forall(var_classes, op.exists(x, P["is_type"].log([x, var_classes])))
        return res
    
    def open_formula(self, *args, **kwargs):
        pass


class Rule_WholeToParts(ltnw.Constraint):
    """Rule that verifies `partOf(x1,x2)` can only be true if the type of x1 and the type of x2 
    validate some rules from wordnet. Here, the rule is about which types can be contained in
    the given type.

    For log-ltn, we reformulate:
        [partof(x1,x2) AND is(x1,whole) ] -> OR[is(x2,part1), ..., is(x2,partn)] 
        <-> Not[partof(x1,x2) AND is(x1,whole)]    OR    OR[is(x2,part1), ..., is(x2,partn)]
        <-> Not[partof(x1,x2)] OR Not[is(x1,whole)]    OR    OR[is(x2,part1), ..., is(x2,partn)]
    """
    def __init__(self, label: str, grounding: ltnw.Grounding, operator_config: ltnw.OperatorConfig, 
            doms_feed_dict: dict[str,ltnw.Domain], whole: str, parts: list[str]) -> None:
        """
        Args:
            whole (str): type of the container
            parts (list[str]): type of the possible parts
        """
        super().__init__(label, grounding, operator_config, doms_feed_dict)
        self.whole = whole
        self.parts = parts
    
    def formula(self, x1: ltn.Variable, x2: ltn.Variable, 
            **kwargs) -> ltn.Formula:
        P, C = self.grounding.predicates, self.grounding.constants
        op = self.operator_config
        not_partof_x1x2 = P["partOf"].nlog([x1,x2])
        not_is_x1_whole = P["is_type"].nlog([x1, C[self.whole]])
        is_x2_parts = [P["is_type"].log([x2, C[part]]) for part in self.parts]

        res = op.or_(op.or_(not_partof_x1x2, not_is_x1_whole),
                    op.or_aggreg(is_x2_parts, broadcast_formulas=True))
        res = op.forall([x1, x2], res)
        return res

    def open_formula(self, *args, **kwargs) -> ltn.Formula:
        return self.formula(*args, keep_formula_open=True, **kwargs)

class Rule_PartToWholes(ltnw.Constraint):
    """Rule that verifies `partOf(x1,x2)` can only be true if the type of x1 and the type of x2 
    validate some rules from wordnet. Here, the rule is about which types can contain the given 
    type.

    For log-ltn, we reformulate:
        [partof(x1,x2) AND is(x1,part) ] -> OR[is(x2,whole1), ..., is(x2,wholen)] 
        <-> Not[partof(x1,x2) AND is(x1,part)]    OR    OR[is(x2,whole1), ..., is(x2,wholen)]
        <-> Not[partof(x1,x2)] OR Not[is(x1,part)]    OR    OR[is(x2,whole1), ..., is(x2,wholen)]
    """
    def __init__(self, label: str, grounding: ltnw.Grounding, operator_config: ltnw.OperatorConfig, 
            doms_feed_dict: dict[str,ltnw.Domain], part: str, wholes: list[str]) -> None:
        """
        Args:
            part (str): type of the part
            wholes (list[str]): type of the possible containers
        """
        super().__init__(label, grounding, operator_config, doms_feed_dict)
        self.part = part
        self.wholes = wholes
    
    def formula(self, x1: ltn.Variable, x2: ltn.Variable, 
            **kwargs) -> ltn.Formula:
        P, C = self.grounding.predicates, self.grounding.constants
        op = self.operator_config
        not_partof_x1x2 = P["partOf"].nlog([x1,x2])
        not_is_x1_part = P["is_type"].nlog([x1, C[self.part]])
        is_x2_wholes = [P["is_type"].log([x2, C[whole]]) for whole in self.wholes]

        res = op.or_(op.or_(not_partof_x1x2, not_is_x1_part),
                        op.or_aggreg(is_x2_wholes, broadcast_formulas=True))

        res = op.forall([x1,x2], res)
        return res

    def open_formula(self, *args, **kwargs) -> ltn.Formula:
        return self.formula(*args, keep_formula_open=True, **kwargs)



class TypeModel(tf.keras.Model):
    """The `TypeModel` class takes in a bounding box and outputs a logits distribution over the
    types of objects that could be in that bounding box
    """
    def __init__(self, n_classes, hidden_layer_sizes = config["types_hidden_layer_sizes"]) -> None:
        super().__init__()
        self.denses = [tf.keras.layers.Dense(s, activation="elu") for s in hidden_layer_sizes]
        self.dense_output = tf.keras.layers.Dense(n_classes)
    
    def call(self, inputs):
        x = inputs[0]
        for dense in self.denses:
            x = dense(x)
        return self.dense_output(x)


class PartOfModel(tf.keras.Model):
    """
    It takes two bounding boxes as input, and outputs a number between 0 and 1, 
    where 0 means the two bounding boxes are not part of each other, and 1 means they are
    """
    def __init__(self, type_model: TypeModel, hidden_layer_sizes = config["partof_hidden_layer_sizes"]) -> None:
        super().__init__()
        self.type_model = type_model
        self.concat = tf.keras.layers.Concatenate()
        self.denses = [tf.keras.layers.Dense(s, activation="elu") for s in hidden_layer_sizes]
        self.dense_output = tf.keras.layers.Dense(1)
    
    def call(self, inputs):
        x1, x2 = inputs[0], inputs[1]
        type_x1, type_x2 = tf.stop_gradient(self.type_model([x1])), tf.stop_gradient(self.type_model([x2]))
        cont_1 = data_processing.containment_ratio_of_bb1_in_bb2(x1, x2, batch_mode=True)
        cont_1 = tf.expand_dims(cont_1, axis=-1)
        cont_2 = data_processing.containment_ratio_of_bb1_in_bb2(x2, x1, batch_mode=True)
        cont_2 = tf.expand_dims(cont_2, axis=-1)
        x1_pos, x2_pos = x1[:,-4:], x2[:, -4:]
        x = self.concat([type_x1,type_x2,x1_pos,x2_pos,cont_1,cont_2])
        for dense in self.denses:
            x = dense(x)
        return self.dense_output(x)

def set_exist_schedule(op_config: ltnw.OperatorConfig) -> None:
    a_min = config["min_a_existential_quantifier"]
    a_max = config["max_a_existential_quantifier"]
    epochs = config["epochs"]
    if config["schedule_p_existential_quantifier"] == "linear":
        a_exists_values = np.linspace(a_min, a_max, epochs)
    elif config["schedule_p_existential_quantifier"] == "square":
        a_exists_values = (np.linspace(a_min, a_max, epochs)**2)*a_max/a_max**2
    elif config["schedule_p_existential_quantifier"] == "exponential":
        a_exists_values = np.exp(np.linspace(a_min, a_max, epochs))*a_max/np.exp(a_max)
    else:
        raise ValueError("`schedule_p_existential_quantifier` in config file is not a valid option.")
    schedule = {ep:val for (ep, val) in zip(range(epochs), a_exists_values)}
    op_config.set_schedule(op_config.exists, "alpha", schedule)
    op_config.set_schedule(op_config.or_aggreg, "alpha", schedule)
    op_config.set_schedule(op_config.or_, "alpha", schedule)
    
def elementwise_operation_on_formula(
        wff: ltn.log.LogFormula, 
        op: Callable[[tf.Tensor],tf.Tensor]
) -> ltn.log.LogFormula:
    result = op(wff.tensor)
    return ltn.log.LogFormula(result, wff.free_vars)