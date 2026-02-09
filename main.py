import polars as pl

from util import make_data, audit_tree, labelencoding#, audit_svc
from ucimlrepo import fetch_ucirepo


SEED = 42
# key = jax.random.PRNGKey(SEED)

def target_enocde(y):
    y_new = y.copy()
    index = 0
    label_to_idx = dict()
    for idx,i in enumerate(y):
        if i.item() not in label_to_idx:
            label_to_idx[i.item()] = index
            index += 1
        
        y_new[idx] = label_to_idx[i.item()]
    return y_new


def main():

    seeds = [ 42,  125,   58,   86,  138,  137,  146,   37,    5,  179,]
    datatypes = ["moons", "circles", 'check']

    res = pl.DataFrame()
    
    for i in datatypes:
        for j in seeds:
            # print(i + f"_{j}")
            # X, y = make_data(5000, .15, i)
            iris = fetch_ucirepo('Adult')
            X = labelencoding(iris.data.features.to_numpy().copy())
            y = labelencoding(iris.data.targets.to_numpy().copy())
            results = audit_tree(X, y,SEED=j, n_splits=10)

            results = results.with_columns(pl.lit(i).alias("distribution"), pl.lit(j).alias("seed"), pl.lit("Tree").alias("model"))
            res = pl.concat([res,results])

            # results = audit_svc(X*10, y,SEED=j, n_splits=10)

            # results = results.with_columns(pl.lit(i).alias("distribution"), pl.lit(j).alias("seed"), pl.lit("SVC").alias("model"))
            # res = pl.concat([res,results])


    
    res.write_parquet("data/test.parquet")


if __name__ == "__main__":
    main()
