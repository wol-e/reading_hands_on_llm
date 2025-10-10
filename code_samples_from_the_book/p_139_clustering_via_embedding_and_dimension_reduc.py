from bertopic import BERTopic
from datasets import load_dataset
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from umap import UMAP

dataset = load_dataset("maartengr/arxiv_nlp")["train"]
abstracts = dataset["Abstracts"]
titles = dataset["Titles"]

#print(titles[0])
#print(abstracts[0])

embedding_model = SentenceTransformer("thenlper/gte-small")
embeddings = embedding_model.encode(abstracts, show_progress_bar=True)

print(embeddings.shape)

# reduce dimensions
umap_model = UMAP(
    n_components=5, min_dist=0.0, metric="cosine", random_state=42
)
reduced_embeddings = umap_model.fit_transform(embeddings)

# cluster
hdbscan_model = HDBSCAN(
    min_cluster_size=50, metric="euclidean", cluster_selection_method="eom"
).fit(reduced_embeddings)
clusters = hdbscan_model.labels_

print(f"Number of found clusters: {len(set(clusters))}")
