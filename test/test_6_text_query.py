import numpy as np

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz

# Step 1: Load your data into FiftyOne
dataset = foz.load_zoo_dataset("quickstart")
model = foz.load_zoo_model("clip-vit-base32-torch")

# Example 4:

fob.compute_similarity(
    dataset,
    model="clip-vit-base32-torch",
    brain_key="lancedb_index",
    backend="lancedb",
)

# Query by vector
query = np.random.rand(512)  # matches the dimension of CLIP embeddings
view = dataset.sort_by_similarity(query, k=10, brain_key="lancedb_index")

# Query by sample ID
query = dataset.first().id
view = dataset.sort_by_similarity(query, k=10, brain_key="lancedb_index")

# Query by a list of IDs
query = [dataset.first().id, dataset.last().id]
view = dataset.sort_by_similarity(query, k=10, brain_key="lancedb_index")

# Query by text prompt
query = "a human with cap"
view = dataset.sort_by_similarity(query, k=10, brain_key="lancedb_index")
session = fo.launch_app(dataset, port=9000)
session.wait()