# Test: patch embeddings

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz

# Step 1: Load your data into FiftyOne
dataset = foz.load_zoo_dataset("quickstart")

# Example 2: reuse embeddings
fob.compute_similarity(
    dataset,
    backend="lancedb",
    brain_key="lancedb_table",
    patches_field="predictions",
)


session = fo.launch_app(dataset, port=9000)
session.wait()
