
# Test: patch embeddings
import numpy as np

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz

# Step 1: Load your data into FiftyOne
dataset = foz.load_zoo_dataset("quickstart")

lancedb_index = fob.compute_similarity(
    dataset,
    brain_key="lancedb_table",
    backend="lancedb",
)
assert lancedb_index.total_index_size == 200  # 200

view = dataset.take(10)
ids = view.values("id")

# Delete 10 samples from a dataset
dataset.delete_samples(view)

# Delete the corresponding vectors from the index
lancedb_index.remove_from_index(sample_ids=ids)
assert lancedb_index.total_index_size == 190  # 190

# Add 20 samples to a dataset
samples = [fo.Sample(filepath="tmp%d.jpg" % i) for i in range(20)]
sample_ids = dataset.add_samples(samples)

# Add corresponding embeddings to the index
embeddings = np.random.rand(20, 1280)
lancedb_index.add_to_index(embeddings, sample_ids)

assert lancedb_index.total_index_size == 210  # 210
session = fo.launch_app(dataset, port=9000)
session.wait()




