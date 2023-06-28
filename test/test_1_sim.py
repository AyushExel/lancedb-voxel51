# TEST: compute similarity and visualize
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob

import numpy as np

dataset = foz.load_zoo_dataset("quickstart")

fob.compute_visualization(dataset, brain_key="img_viz")

fob.compute_similarity(
    dataset,
    backend="lancedb", brain_key="lancedb_index", metric="l2"
)

fob.compute_similarity(
    dataset,
    backend="lancedb", brain_key="lancedb_index_cos", metric="cosine"
)


session = fo.launch_app(dataset, port=9000)
session.wait()
