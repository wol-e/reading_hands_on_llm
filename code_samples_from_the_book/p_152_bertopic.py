import pandas as pd

from bertopic import BERTopic
from datasets import load_dataset
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from umap import UMAP

dataset = load_dataset("maartengr/arxiv_nlp")["train"]
abstracts = dataset["Abstracts"]
titles = dataset["Titles"]

embedding_model = SentenceTransformer("thenlper/gte-small")
embeddings = embedding_model.encode(abstracts, show_progress_bar=True)
umap_model = UMAP(
    n_components=5, min_dist=0.0, metric="cosine", random_state=42
)
hdbscan_model = HDBSCAN(
    min_cluster_size=50, metric="euclidean", cluster_selection_method="eom"
)

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    verbose=True
).fit(abstracts, embeddings)

pd.options.display.max_columns = 10
print(topic_model.get_topic_info())
