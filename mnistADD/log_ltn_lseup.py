import argparse
import tensorflow as tf
import ltn
import baselines, data, commons
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--csv-path',type=str,default="MDadd_log_ltn_lseup.csv")
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--n-examples-train',type=int,default=15000)
    parser.add_argument('--n-examples-test',type=int,default=2500)
    parser.add_argument('--batch-size',type=int,default=32)
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args



args = parse_args()
n_examples_train = args['n_examples_train']
n_examples_test = args['n_examples_test']
batch_size = args['batch_size']
EPOCHS = args['epochs']
csv_path = args['csv_path']

""" DATASET """

ds_train, ds_test = data.get_mnist_op_dataset(
        count_train=n_examples_train,
        count_test=n_examples_test,
        buffer_size=10000,
        batch_size=batch_size,
        n_operands=4,
        op=lambda args: 10*args[0]+args[1]+10*args[2]+args[3])

""" LTN MODEL AND LOSS """
### Predicates
logits_model = baselines.SingleDigit(inputs_as_a_list=True)
Digit = ltn.log.Predicate.FromLogits(logits_model,activation_function="softmax")
### Variables
d1 = ltn.Variable("digits1", range(10))
d2 = ltn.Variable("digits2", range(10))
d3 = ltn.Variable("digits3", range(10))
d4 = ltn.Variable("digits4", range(10))
### Operators
And = ltn.log.Wrapper_Connective(ltn.log.fuzzy_ops.And_Sum())
Or = ltn.log.Wrapper_Connective(ltn.log.fuzzy_ops.Or_LogSumExp(alpha=1))
Forall = ltn.log.Wrapper_Quantifier(ltn.log.fuzzy_ops.Aggreg_Sum(), semantics="forall")
Exists = ltn.log.Wrapper_Quantifier(ltn.log.fuzzy_ops.Aggreg_LogSumExpUpperBound(alpha=1),semantics="exists")
formula_aggregator = ltn.log.Wrapper_Formula_Aggregator(ltn.log.fuzzy_ops.Aggreg_Sum())

# mask
add = ltn.Function.Lambda(lambda inputs: inputs[0]+inputs[1])
times = ltn.Function.Lambda(lambda inputs: inputs[0]*inputs[1])
ten = ltn.Constant(10, trainable=False)
equals = ltn.Predicate.Lambda(lambda inputs: inputs[0] == inputs[1])
two_digit_number = lambda inputs : add([times([ten,inputs[0]]), inputs[1] ])

#@tf.function
def axioms(images_x1,images_x2,images_y1,images_y2,labels_z,alpha_exists):
    images_x1 = ltn.Variable("x1", images_x1)
    images_x2 = ltn.Variable("x2", images_x2)
    images_y1 = ltn.Variable("y1", images_y1)
    images_y2 = ltn.Variable("y2", images_y2)
    labels_z = ltn.Variable("z", labels_z)
    axiom = Forall(
            ltn.diag(images_x1,images_x2,images_y1,images_y2,labels_z),
            Exists(
                (d1,d2,d3,d4),
                And(
                    And(Digit.log([images_x1,d1]),Digit.log([images_x2,d2])),
                    And(Digit.log([images_y1,d3]),Digit.log([images_y2,d4]))
                ),
                mask=equals([labels_z, add([ two_digit_number([d1,d2]), two_digit_number([d3,d4]) ]) ]),
                alpha=alpha_exists
            )
        )
    sat = axiom.tensor
    return sat
### Initialize layers and weights
x1, x2, y1, y2, z = next(ds_train.as_numpy_iterator())
axioms(x1, x2, y1, y2, z, 2)

""" TRAINING """

optimizer = tf.keras.optimizers.Adam(0.001)
metrics_dict = {
    'train_loss': tf.keras.metrics.Mean(name="train_loss"),
    'train_accuracy': tf.keras.metrics.Mean(name="train_accuracy"),
    'test_loss': tf.keras.metrics.Mean(name="test_loss"),
    'test_accuracy': tf.keras.metrics.Mean(name="test_accuracy")    
}

#@tf.function
def train_step(images_x1,images_x2,images_y1,images_y2,labels_z,**kwargs):
    # loss
    with tf.GradientTape() as tape:
        loss = - axioms(images_x1,images_x2,images_y1,images_y2,labels_z,**kwargs)
    gradients = tape.gradient(loss, logits_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, logits_model.trainable_variables))
    metrics_dict['train_loss'](loss)
    # accuracy
    predictions_x1 = tf.argmax(logits_model([images_x1]),axis=-1)
    predictions_x2 = tf.argmax(logits_model([images_x2]),axis=-1)
    predictions_y1 = tf.argmax(logits_model([images_y1]),axis=-1)
    predictions_y2 = tf.argmax(logits_model([images_y2]),axis=-1)
    predictions_z = 10*predictions_x1+predictions_x2+10*predictions_y1+predictions_y2
    match = tf.equal(predictions_z,tf.cast(labels_z,predictions_z.dtype))
    metrics_dict['train_accuracy'](tf.reduce_mean(tf.cast(match,tf.float32)))
    
#@tf.function
def test_step(images_x1,images_x2,images_y1,images_y2,labels_z,**kwargs):
    # loss
    loss = - axioms(images_x1,images_x2,images_y1,images_y2,labels_z,**kwargs)
    metrics_dict['test_loss'](loss)
    # accuracy
    predictions_x1 = tf.argmax(logits_model([images_x1]),axis=-1)
    predictions_x2 = tf.argmax(logits_model([images_x2]),axis=-1)
    predictions_y1 = tf.argmax(logits_model([images_y1]),axis=-1)
    predictions_y2 = tf.argmax(logits_model([images_y2]),axis=-1)
    predictions_z = 10*predictions_x1+predictions_x2+10*predictions_y1+predictions_y2
    match = tf.equal(predictions_z,tf.cast(labels_z,predictions_z.dtype))
    metrics_dict['test_accuracy'](tf.reduce_mean(tf.cast(match,tf.float32)))

from collections import defaultdict


scheduled_parameters = defaultdict(lambda: {})
a_min = 1
a_max = 5
a_exists_values = np.linspace(a_min, a_max, EPOCHS, dtype=np.float32)
for epoch in range(EPOCHS):
    scheduled_parameters[epoch] = {"alpha_exists": tf.constant(a_exists_values[epoch])}

commons.train(
    EPOCHS,
    metrics_dict,
    ds_train,
    ds_test,
    train_step,
    test_step,
    csv_path=csv_path,
    scheduled_parameters=scheduled_parameters
)