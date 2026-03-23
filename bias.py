import numpy as np
import polars as pl
from ucimlrepo import fetch_ucirepo

from util import audit_tree_bias, labelencoding  # , audit_svc

SEED = 42
# key = jax.random.PRNGKey(SEED)

# remove normal distribution around some point for bias
# more intresting bias types bias selection
# katerina dost bias detection
# stats tests


def main():

    seeds = [
        42,
        125,
        58,
        86,
        138,
        137,
        146,
        37,
        5,
        179,
    ]
    # datatypes = ["moons", "circles"] # , 'check'
    introduced_bias = np.linspace(0.1, 1, 9, False)

    res = pl.DataFrame()

    for dataset in ["iris", "wine", "Car Evaluation"]:
        data = fetch_ucirepo(dataset)
        X = labelencoding(data.data.features.to_numpy().copy())
        y = labelencoding(data.data.targets.to_numpy().copy())

        for i in np.unique(y):
            for feature in range(X.shape[1]):
                for j in seeds:
                    for k in introduced_bias:
                        # print(i + f"_{j}")
                        # X, y = make_data(5000, .15, i)
                        results = audit_tree_bias(
                            X,
                            y,
                            k,
                            stopping=feature,
                            SEED=j,
                            n_splits=10,
                            dbscan=True,
                            target_bias=i.item(),
                        )

                        results = results.with_columns(
                            pl.lit(i).alias("distribution"),
                            pl.lit(j).alias("seed"),
                            pl.lit(dataset + "_feature" + str(feature)).alias(
                                "dataset"
                            ),
                            pl.lit("undersampling").alias("bias_type"),
                        )
                        res = pl.concat([res, results])

                        results = audit_tree_bias(
                            X,
                            y,
                            k,
                            stopping=feature,
                            SEED=j,
                            n_splits=10,
                            dbscan=True,
                            target_bias=i.item(),
                            over=True,
                        )

                        results = results.with_columns(
                            pl.lit(i).alias("distribution"),
                            pl.lit(j).alias("seed"),
                            pl.lit(dataset + "_feature" + str(feature)).alias(
                                "dataset"
                            ),
                            pl.lit("oversampling").alias("bias_type"),
                        )
                        res = pl.concat([res, results])

            # results = audit_svc(X*10, y,SEED=j, n_splits=10)

            # results = results.with_columns(pl.lit(i).alias("distribution"), pl.lit(j).alias("seed"), pl.lit("SVC").alias("model"))
            # res = pl.concat([res,results])

    res.write_parquet("data/data_bias.parquet")


if __name__ == "__main__":
    main()
