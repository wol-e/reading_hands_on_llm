import numpy as np
import pandas as pd

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset

# Load our data
data = load_dataset("rotten_tomatoes")

print(data["train"][0, -1])


## first approach: using a task specific model, i.e. roberta finetuned for sentiment analysis on tweets
model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Load model into pipeline
pipe = pipeline(
    model=model_path,
    tokenizer=model_path,
    return_all_scores=True,
    device="cuda:0"
)

# Run inference
y_pred = []
for output in tqdm(pipe(KeyDataset(data["test"], "text")), total=len(data["test"])):
    negative_score = output[0]["score"]
    positive_score = output[2]["score"]
    assignment = np.argmax([negative_score, positive_score])
    y_pred.append(assignment)

def evaluate_performance(y_true, y_pred):
    """Create and print the classification report"""
    performance = classification_report(
        y_true, y_pred,
        target_names=["Negative Review", "Positive Review"]
    )
    print(performance)

print("Performance of twitter-roberta-base-sentiment-latest")
evaluate_performance(data["test"]["label"], y_pred)

## second approach: use a logistic regression on features obtained via embedding
# Load model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Convert text to embeddings
train_embeddings = model.encode(data["train"]["text"], show_progress_bar=True)
test_embeddings = model.encode(data["test"]["text"], show_progress_bar=True)

clf = LogisticRegression(random_state=42)
clf.fit(train_embeddings, data["train"]["label"])

y_pred = clf.predict(test_embeddings)

print("Performance of logistic regression on embedded features")
evaluate_performance(data["test"]["label"], y_pred)

# Approach 3, Using cosine similarity of embeddings
# Average the embeddings of all documents in each target label
df = pd.DataFrame(np.hstack([train_embeddings, np.array(data["train"]["label"]).reshape(-1, 1)]))
averaged_target_embeddings = df.groupby(768).mean().values

# Find the best matching embeddings between evaluation documents and target embeddings
sim_matrix = cosine_similarity(test_embeddings, averaged_target_embeddings)
y_pred = np.argmax(sim_matrix, axis=1)

# Evaluate the model
print("Performance using cosine similarity")
evaluate_performance(data["test"]["label"], y_pred)

# Approach 4: Zero shot classification
label_embeddings = model.encode(["A negative review",  "A positive review"])

sim_matrix = cosine_similarity(test_embeddings, label_embeddings)
y_pred = np.argmax(sim_matrix, axis=1)

print("Zero Shot classification performance")
evaluate_performance(data["test"]["label"], y_pred)
