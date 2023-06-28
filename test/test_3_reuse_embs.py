# Test: Reuse embeddings

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz

# Step 1: Load your data into FiftyOne
dataset = foz.load_zoo_dataset("quickstart")

# Example 2: reuse embeddings
fob.compute_similarity(
    dataset,
    backend="lancedb",
    table_name="test_table_5",
)


fob.compute_similarity(
    dataset,
    embeddings=False,                   # don't compute embeddings
    table_name="test_table_5",            # the existing lancedb index
    brain_key="lancedb_table_sim",
    backend="lancedb",
)

session = fo.launch_app(dataset, port=9000)
session.wait()

