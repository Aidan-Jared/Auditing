import argparse as parser
import polars as pl

from util import make_data, audit_tree_bias#, audit_svc

import jax
import jax.numpy as jnp
from jaxtyping import PyTree,  Array
import equinox as eqx
import numpy as np

SEED = 42
# key = jax.random.PRNGKey(SEED)

def main():

    seeds = [42,  125,   58,   86,  138,  137,  146,   37,    5,  179,]
    datatypes = ["moons", "circles"] # , 'check'
    introduced_bias = [3.,2.5,2.25,2.,1.75,1.5,1.25,1.,.75,.5,.25,0.]

    res = pl.DataFrame()
    
    for i in datatypes:
        for j in seeds:
            for k in introduced_bias:
                print(i + f"_{j}")
                X, y = make_data(5000, .15, i)
                

                results = audit_tree_bias(X, y, k,SEED=j, n_splits=10)

                results = results.with_columns(pl.lit(i).alias("distribution"), pl.lit(j).alias("seed"), pl.lit("Tree").alias("model"))
                res = pl.concat([res,results])

            # results = audit_svc(X*10, y,SEED=j, n_splits=10)

            # results = results.with_columns(pl.lit(i).alias("distribution"), pl.lit(j).alias("seed"), pl.lit("SVC").alias("model"))
            # res = pl.concat([res,results])


    
    res.write_parquet("data/data_bias.parquet")


if __name__ == "__main__":
    main()
