# Test: Query from index

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz

# Step 1: Load your data into FiftyOne
dataset = foz.load_zoo_dataset("quickstart")

# Steps 2 and 3: Compute embeddings and create a similarity index
brain_key = "lancedb_index"
lancedb_index = fob.compute_similarity(
    dataset,
    brain_key=brain_key,
    backend="lancedb",
)

# Step 4: Query your data
query = dataset.last().id  # query by sample ID
view = dataset.sort_by_similarity(
    query,
    brain_key=brain_key,
    k=50,  # limit to 50 most similar samples
)

# Step 4_2: Query your data with a list
query = dataset.take(10).values("id")  # query by sample ID 
view = dataset.sort_by_similarity(
    query,
    brain_key=brain_key,
    k=50,  # limit to 50 most similar samples
)

# Step 5 (optional): Cleanup

# Delete the Lancedb index
lancedb_index = dataset.load_brain_results(brain_key)
lancedb_index.cleanup()

# Delete run record from FiftyOne
dataset.delete_brain_run("lancedb_index")

session = fo.launch_app(view, port=9000)
session.wait()
