import tensorflow as tf
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import ltn

import data

# DATA
dataset = data.load_pca_data("TCGA-PANCAN-HiSeq-801x20531/pca_16d.csv")
nr_of_clusters = dataset.nb_clusters
features = dataset.features

distances = euclidean_distances(features, features)
close_threshold = np.percentile(distances, 5)

# GROUNDING
class MLP(tf.keras.Model):
    """ Model to call as P(x,class) """
    def __init__(self, n_classes, hidden_layer_sizes=(16,16,16)):
        super().__init__()
        self.denses = [tf.keras.layers.Dense(s, activation="elu") for s in hidden_layer_sizes]
        self.dense_class = tf.keras.layers.Dense(n_classes)
        
    def call(self, inputs):
        x = inputs[0]
        for dense in self.denses:
            x = dense(x)
        logits = self.dense_class(x)
        return logits

logits_model = MLP(nr_of_clusters)
C = ltn.log.Predicate.FromLogits(logits_model, activation_function="softmax", with_class_indexing=True)
cluster = ltn.Variable("cluster",range(nr_of_clusters))

x = ltn.Variable("x",features)
y = ltn.Variable("y",features)

And = ltn.log.Wrapper_Connective(ltn.log.fuzzy_ops.And_Sum())
Or = ltn.log.Wrapper_Connective(ltn.log.fuzzy_ops.Or_LogSumExp(alpha=5))
Forall = ltn.log.Wrapper_Quantifier(ltn.log.fuzzy_ops.Aggreg_Mean(), semantics="forall")
Exists = ltn.log.Wrapper_Quantifier(ltn.log.fuzzy_ops.Aggreg_LogSumExp(alpha=5),semantics="exists")
formula_aggregator = ltn.log.Wrapper_Formula_Aggregator(ltn.log.fuzzy_ops.Aggreg_Sum())

eucl_dist = ltn.Function.Lambda(lambda inputs: tf.expand_dims(tf.norm(inputs[0]-inputs[1],axis=1),axis=1))
is_greater_than = ltn.Predicate.Lambda(lambda inputs: inputs[0] > inputs[1])
close_thr = ltn.Constant(close_threshold, trainable=False)

# CONSTRAINTS
def axioms(alpha_exists):
    axioms = [
        Forall(x, Exists(cluster, C.log([x, cluster]), alpha=alpha_exists)),
        Forall(cluster, Exists(x, C.log([x, cluster]), alpha=alpha_exists)),
        Forall([cluster,x,y], 
                Or(C.nlog([x,cluster]),C.log([y,cluster]), alpha=alpha_exists),
                mask = is_greater_than([close_thr,eucl_dist([x,y])])),
    ]
    
    sat_level = formula_aggregator(axioms).tensor
    return sat_level

axioms(alpha_exists=5) # first call to build the graph

# TRAINING

trainable_variables = logits_model.trainable_variables
optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)

epochs = 1000
epochs_fixed_schedule = 0
alpha_exists = np.concatenate([
        [1]*epochs_fixed_schedule,
        np.linspace(1, 4, epochs-epochs_fixed_schedule)])
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss = - axioms(alpha_exists[epoch])
    grads = tape.gradient(loss, logits_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, logits_model.trainable_variables))
    if epoch%100 == 0:
        log_sat = axioms(alpha_exists[epoch])
        loss = - log_sat
        sat = tf.math.exp(log_sat)
        print("Epoch %d: Sat Level %.3f, Loss %.3f"%(epoch, sat, loss))
log_sat = axioms(alpha_exists[epoch])
loss = - log_sat
sat = tf.math.exp(log_sat)
print("Training finished at Epoch %d with Sat Level %.3f, Loss %.3f"%(epoch, sat, loss))

# EVALUATE
predictions = tf.math.argmax(C([x,cluster]).tensor,axis=1)
print(data.adjusted_rand_score(dataset.labels,predictions)) # 0.63
data.save_pdf_predictions(features, predictions, dataset.label_names, save_prefix="logltn")