import os
import logging
import numpy as np
import fiftyone.core.utils as fou
from fiftyone.brain.similarity import (
    SimilarityConfig,
    Similarity,
    SimilarityIndex,
)
import eta.core.utils as etau

import fiftyone.brain.internal.core.utils as fbu

lancedb = fou.lazy_import("lancedb")
pa = fou.lazy_import("pyarrow")
pd = fou.lazy_import("pandas")

_SUPPORTED_METRICS = ("cosine", "l2")

logger = logging.getLogger(__name__)

class LanceDBSimilarityConfig(SimilarityConfig):
    def __init__(
        self,
        embeddings_field=None,
        model=None,
        patches_field=None,
        supports_prompts=None,
        table_name=None,
        metric=None,
        uri="/tmp/lancedb",
        **kwargs,
    ):
        if metric is not None and metric not in _SUPPORTED_METRICS:
            raise ValueError(
                "Unsupported metric '%s'. Supported values are %s"
                % (metric, _SUPPORTED_METRICS)
            )
        super().__init__(
            embeddings_field=embeddings_field,
            model=model,
            patches_field=patches_field,
            supports_prompts=supports_prompts,
            **kwargs,
        )
        self.table_name = table_name
        self.uri = os.path.abspath(uri)
        self.metric = metric

    @property
    def method(self):
        """The name of the similarity backend."""
        return "lancedb"

    @property
    def max_k(self):
        """A maximum k value for nearest neighbor queries, or None if there is
        no limit.
        """
        return None # TODO: check this

    @property
    def supports_least_similarity(self):
        """Whether this backend supports least similarity queries."""
        return False

    @property
    def supported_aggregations(self):
        return ("mean",)

class LanceDBSimilarity(Similarity):
    """LanceDB similarity factory.

    Args:
        config: a :class:`LanceDBSimilarityConfig`
    """

    def ensure_requirements(self):
        fou.ensure_package("lancedb")

    def ensure_usage_requirements(self):
        fou.ensure_package("lancedb")

    def initialize(self, samples, brain_key):
        return LanceDBSimilarityIndex(
            samples, self.config, brain_key, backend=self
        )

class LanceDBSimilarityIndex(SimilarityIndex):
    def __init__(self, samples, config, brain_key, backend=None):
        super().__init__(samples, config, brain_key, backend=backend)
        self._table = None
        self._db = None
        self._initialize()

    def _initialize(self):
        db = lancedb.connect(self.config.uri)
        tables = db.table_names()

        if self.config.table_name is None:
            root = "fiftyone-" + fou.to_slug(self.samples._root_dataset.name)
            table_name = fbu.get_unique_name(root, tables)

            self.config.table_name = table_name
            self.save_config()

        if self.config.table_name in tables:
            table = db.open_table(self.config.table_name)
        else:
            table = None
        self._db = db
        self._table = table
        self._table_name = self.config.table_name
    
    @property
    def table(self):
        """The ``lancedb.LanceTable`` instance for this table."""
        return self._table

    @property
    def total_index_size(self):
        if self._table is None:
            return None

        return len(self._table)
    
    def _get_pa_table(self):
        if self._table is not None:
            return self._table.to_arrow()

        return pa.Table.from_arrays([[],[], []], names=["id", "sample_id", "vector"]) # TODO: make names variable

    def add_to_index(
        self,
        embeddings,
        sample_ids,
        label_ids=None,
        overwrite=True,
        allow_existing=True,
        warn_existing=False,
        reload=True,
    ):
        """Adds the given embeddings to the index.

        Args:
            embeddings: a ``num_embeddings x num_dims`` array of embeddings
            sample_ids: a ``num_embeddings`` array of sample IDs
            label_ids (None): a ``num_embeddings`` array of label IDs, if
                applicable
            overwrite (True): whether to replace (True) or ignore (False)
                existing embeddings with the same sample/label IDs
            allow_existing (True): whether to ignore (True) or raise an error
                (False) when ``overwrite`` is False and a provided ID already
                exists in the
            warn_missing (False): whether to log a warning if an embedding is
                not added to the index because its ID already exists
            reload (True): whether to call :meth:`reload` to refresh the
                current view after the update
        """
        pa_table = self._get_pa_table()

        if label_ids is not None:
            ids = label_ids
        else:
            ids = sample_ids
        
        if warn_existing or not allow_existing or not overwrite:
            existing_ids = set(pa_table["id"].to_pylist()) & set(ids)
            num_existing = len(existing_ids)

            if num_existing > 0:
                if not allow_existing:
                    raise ValueError(
                        "Found %d IDs (eg %s) that already exist in the index"
                        % (num_existing, next(iter(existing_ids)))
                    )

                if warn_existing:
                    if overwrite:
                        logger.warning(
                            "Overwriting %d IDs that already exist in the "
                            "index",
                            num_existing,
                        )
                    else:
                        logger.warning(
                            "Skipping %d IDs that already exist in the index",
                            num_existing,
                        )
        else:
            existing_ids = set()

        if existing_ids and not overwrite:
            del_inds = [i for i, _id in enumerate(ids) if _id in existing_ids]

            embeddings = np.delete(embeddings, del_inds)
            sample_ids = np.delete(sample_ids, del_inds)
            if label_ids is not None:
                label_ids = np.delete(label_ids, del_inds)
        
        if label_ids is not None:
            ids = list(label_ids)
        else:
            ids = list(sample_ids)

        # TODO: simpler to do with pandas integration.        
        len = self.total_index_size
        dim = embeddings.shape[1]
        if self._table: # update the table
            prev_embeddings = np.concatenate(pa_table["vector"].to_numpy()).reshape(-1, dim)
            embeddings = np.concatenate([prev_embeddings, embeddings])
            ids = pa_table["id"].to_pylist() + ids
            sample_ids = pa_table["sample_id"].to_pylist() + sample_ids

        embeddings = pa.array(embeddings.reshape(-1), type=pa.float32())
        embeddings = pa.FixedSizeListArray.from_arrays(embeddings, dim)
        sample_ids = list(sample_ids)
        pa_table = pa.Table.from_arrays([ids,sample_ids, embeddings], names=["id", "sample_id", "vector"])
        self._table = self._db.create_table(self.config.table_name, pa_table, mode="overwrite")

        if reload:
            self.reload()

    def remove_from_index(
        self,
        sample_ids=None,
        label_ids=None,
        allow_missing=True,
        warn_missing=False,
        reload=True,
    ):
        if label_ids is not None:
            ids = label_ids
        else:
            ids = sample_ids

        if not allow_missing or warn_missing:
            existing_ids = self._index.fetch(ids).vectors.keys()
            missing_ids = set(existing_ids) - set(ids)
            num_missing = len(missing_ids)

            if num_missing > 0:
                if not allow_missing:
                    raise ValueError(
                        "Found %d IDs (eg %s) that are not present in the "
                        "index" % (num_missing, missing_ids[0])
                    )

                if warn_missing:
                    logger.warning(
                        "Ignoring %d IDs that are not present in the index",
                        num_missing,
                    )

        df = self._table.to_pandas()
        df = df[~df["id"].isin(ids)]
        self._table = self._db.create_table(self.config.table_name, df, mode="overwrite")

        if reload:
            self.reload()

    def get_embeddings(
        self,
        sample_ids=None,
        label_ids=None,
        allow_missing=True,
        warn_missing=False,
    ):
        """Retrieves the embeddings for the given IDs from the index.

        If no IDs are provided, the entire index is returned.

        Args:
            sample_ids (None): a sample ID or list of sample IDs for which to
                retrieve embeddings
            label_ids (None): a label ID or list of label IDs for which to
                retrieve embeddings
            allow_missing (True): whether to allow the index to not contain IDs
                that you provide (True) or whether to raise an error in this
                case (False)
            warn_missing (False): whether to log a warning if the index does
                not contain IDs that you provide

        Returns:
            a tuple of:

            -   a ``num_embeddings x num_dims`` array of embeddings
            -   a ``num_embeddings`` array of sample IDs
            -   a ``num_embeddings`` array of label IDs, if applicable, or else
                ``None``
        """
        if label_ids is not None:
            if self.config.patches_field is None:
                raise ValueError("This index does not support label IDs")

            if sample_ids is not None:
                logger.warning(
                    "Ignoring sample IDs when label IDs are provided"
                )

        pd_table = self._table.to_pandas()
        found_embeddings, found_sample_ids, found_label_ids, missing_ids = [], [], [], []
        if sample_ids is not None and self.config.patches_field is not None:
            sample_ids = sample_ids if isinstance(sample_ids, list) else [sample_ids]
            df = pd_table.set_index("sample_id")
            for sample_id in sample_ids:
                if sample_id in df.index:
                    found_embeddings.append(df.loc[sample_id]["vector"])
                    found_sample_ids.append(sample_id)
                    found_label_ids.append(df.loc[sample_id]["id"])
                else:
                    missing_ids.append(sample_id)

        elif self.config.patches_field is not None:
            df = pd_table.set_index("id")
            label_ids = label_ids if isinstance(label_ids, list) else [label_ids]
            for label_id in label_ids:
                if label_id in df.index:
                    found_embeddings.append(df.loc[label_id]["vector"])
                    found_sample_ids.append(df.loc[label_id]["sample_id"])
                    found_label_ids.append(label_id)
                else:
                    missing_ids.append(label_id)
        else:
            df = pd_table.set_index("sample_id")
            sample_id = sample_ids if isinstance(sample_ids, list) else [sample_ids]
            for sample_id in sample_ids:
                if sample_id in df.index:
                    found_embeddings.append(df.loc[sample_id]["vector"])
                    found_sample_ids.append(sample_id)
                else:
                    missing_ids.append(sample_id)

        num_missing_ids = len(missing_ids)
        if num_missing_ids > 0:
            if not allow_missing:
                raise ValueError(
                    "Found %d IDs (eg %s) that do not exist in the index"
                    % (num_missing_ids, missing_ids[0])
                )

            if warn_missing:
                logger.warning(
                    "Skipping %d IDs that do not exist in the index",
                    num_missing_ids,
                )
        
        embeddings = np.array(found_embeddings)
        sample_ids = np.array(found_sample_ids)
        if label_ids is not None:
            label_ids = np.array(found_label_ids)

        return embeddings, sample_ids, label_ids

    def cleanup(self):
        if self._db is not None:
            self._db.drop_table(self.config.table_name)
            self._table = None

    def _kneighbors(
        self,
        query=None,
        k=None,
        reverse=False,
        aggregation=None,
        return_dists=False,
    ):
        if query is None:
            raise ValueError("LanceDB does not support full index neighbors") # TODO: check this

        if aggregation not in (None, "mean"):
            raise ValueError(f"LanceDB does not support {aggregation} aggregation") # TODO: check this

        if k is None:
            k = len(self._table.to_arrow())

        query = self._parse_neighbors_query(query)
        if aggregation == "mean" and query.ndim == 2:
            query = query.mean(axis=0)

        single_query = query.ndim == 1
        if single_query:
            query = [query]

        if self.config.patches_field is not None:
            index_ids = self.current_label_ids
        else:
            index_ids = self.current_sample_ids

        ids = []
        dists = []
        for q in query:
            #import pdb;pdb.set_trace()
            results = self._table.search(q)
            if self.config.metric is not None:
                results = results.metric(self.config.metric)

            results = results.limit(k).to_df().query("id in @index_ids")
            if reverse:
                results = results.iloc[::-1]

            ids.append(results.id.tolist())
            if return_dists:
                dists.append(results.score.tolist())

        if single_query:
            ids = ids[0]
            if return_dists:
                dists = dists[0]
        if return_dists:
            return ids, dists

        return ids

    def _parse_neighbors_query(self, query):
        if etau.is_str(query):
            query_ids = [query]
            single_query = True
        else:
            query = np.asarray(query)

            # Query by vector(s)
            if np.issubdtype(query.dtype, np.number):
                return query

            query_ids = list(query)
            single_query = False

        # Query by ID(s)
        table = self._db.open_table(self._table_name)
        
        embeddings = table.to_pandas().set_index("id").loc[query_ids]["vector"]
        query = np.array([emb for emb in embeddings])

        if single_query:
            query = query[0, :]

        return query
    
    @classmethod
    def _from_dict(cls, d, samples, config, brain_key):
        return cls(samples, config, brain_key)
