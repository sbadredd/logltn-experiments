from dataclasses import dataclass
import datetime
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import adjusted_rand_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

@dataclass
class TCGA_DATASET:
    features: np.ndarray
    labels: np.ndarray
    label_encoder: LabelEncoder

    @property
    def nb_clusters(self) -> int:
        return len(self.label_encoder.classes_)
    
    @property
    def label_names(self) -> np.ndarray:
        return self.label_encoder.inverse_transform(self.labels)

def load_data() -> TCGA_DATASET:
    """
    It loads the data from the files, and returns a `TCGA_DATASET` object
    
    Returns:
      A TCGA_DATASET object with three fields: features, labels, and label_encoder.
    """
    features_file = "TCGA-PANCAN-HiSeq-801x20531/data.csv"
    labels_file = "TCGA-PANCAN-HiSeq-801x20531/labels.csv"
    features = np.genfromtxt(features_file, delimiter=",", usecols=range(1, 20532), skip_header=1)
    label_names = np.genfromtxt(labels_file, delimiter=",", usecols=(1,), skip_header=1, dtype="str")
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(label_names)
    return TCGA_DATASET(features, labels, label_encoder)

def load_pca_data(features_file: str) -> TCGA_DATASET:
    features = np.genfromtxt(features_file, delimiter=",", skip_header=1)
    labels_file = "TCGA-PANCAN-HiSeq-801x20531/labels.csv"
    label_names = np.genfromtxt(labels_file, delimiter=",", usecols=(1,), skip_header=1, dtype="str")
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(label_names)
    return TCGA_DATASET(features, labels, label_encoder)

def preprocess_features(features: np.ndarray, pca_components: int) -> np.ndarray:
    """
    Takes in an array of feature vectors, normalize them with MinMax and reduce their dimensionality
    with PCA.
 
    Args:
      features (np.ndarray): the features to be preprocessed
      pca_components (int): The number of components to keep after PCA.
    
    Returns:
      The new features are being returned.
    """
    preprocessor = Pipeline([
            ("scaler", MinMaxScaler()),
            ("pca", PCA(n_components=pca_components, random_state=0)),
    ])
    new_features = preprocessor.fit_transform(features)
    return new_features

def calculate_rand_index(true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    """Calculate the rand index adjusted for chance.  

    Args:
        true_labels (np.ndarray): ground truth cluster labels.
        predicted_labels (np.ndarray): cluster labels to evaluate.

    Returns:
        float: ARI score.
    """
    return adjusted_rand_score(true_labels, predicted_labels)

def save_pdf_predictions(
        pca_features: np.ndarray,
        predicted_labels: np.ndarray,
        true_label_names: np.ndarray,
        save_prefix: str = ""
) -> None:
    pcadf = pd.DataFrame(pca_features, columns=[f"component_{i}" for i in range(pca_features.shape[1])])
    pcadf["predicted_cluster"] = predicted_labels
    pcadf["true_label"] = true_label_names
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    pcadf.to_csv(save_prefix+current_time+".csv", index=False, sep=",")

def plot_predictions(
        pca_features: np.ndarray, 
        predicted_labels: np.ndarray, 
        true_label_names: np.ndarray):
    pcadf = pd.DataFrame(pca_features, columns=["component_1", "component_2"])
    pcadf["predicted_cluster"] = predicted_labels
    pcadf["true_label"] = true_label_names

    plt.style.use("fivethirtyeight")
    plt.figure(figsize=(8, 8))

    scat = sns.scatterplot(
        data=pcadf,
        x="component_1",
        y="component_2",
        s=50,
        hue="predicted_cluster",
        style="true_label",
        palette="Set2",
    )

    scat.set_title(
        "Clustering results from TCGA Pan-Cancer\nGene Expression Data"
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    plt.show()
    return 0