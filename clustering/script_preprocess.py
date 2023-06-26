import data
import pandas as pd

pca_components = 16

dataset = data.load_data()
pca_features = data.preprocess_features(dataset.features, pca_components=pca_components)
pcadf = pd.DataFrame(pca_features, columns=[f"component_{i}" for i in range(pca_components)])
pcadf.to_csv(f"pca_{pca_components}d.csv", sep=",", index=False)