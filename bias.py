import argparse as parser
import polars as pl

from util import make_data, audit_tree_bias, labelencoding#, audit_svc
from ucimlrepo import fetch_ucirepo
import numpy as np

SEED = 42
# key = jax.random.PRNGKey(SEED)

def main():

    seeds = [42,  125,   58,   86,  138,  137,  146,   37,    5,  179,]
    # datatypes = ["moons", "circles"] # , 'check'
    introduced_bias = np.linspace(.1,1, 9,False)

    data = fetch_ucirepo('iris')
    X = labelencoding(data.data.features.to_numpy().copy())
    y = labelencoding(data.data.targets.to_numpy().copy())

    res = pl.DataFrame()
    
    for i in np.unique(y):
        for j in seeds:
            for k in introduced_bias:
                # print(i + f"_{j}")
                # X, y = make_data(5000, .15, i)
                results = audit_tree_bias(X, y, k,SEED=j, n_splits=10, dbscan=True, target_bias=i.item())

                results = results.with_columns(pl.lit(i).alias("distribution"), pl.lit(j).alias("seed"), pl.lit("Tree").alias("model"))
                res = pl.concat([res,results])

            # results = audit_svc(X*10, y,SEED=j, n_splits=10)

            # results = results.with_columns(pl.lit(i).alias("distribution"), pl.lit(j).alias("seed"), pl.lit("SVC").alias("model"))
            # res = pl.concat([res,results])


    
    res.write_parquet("data/data_bias.parquet")


if __name__ == "__main__":
    main()
