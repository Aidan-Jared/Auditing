import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np

from jaxtyping import PyTree, Array, Int, Float
from jax_meta.utils.losses import cross_entropy
from jax_meta.utils.metrics import accuracy

from torch.utils.data import DataLoader, TensorDataset
from torch import manual_seed, tensor

from util import make_data, FGSM

import pandas as pd

import tqdm

SEED = 42
BATCH_SIZE = 32
EPOCHS = int(1e3)
KEY = jax.random.PRNGKey(SEED)
manual_seed(SEED)

# @eqx.filter_jit
def find_distance(
        model: PyTree,
        X: Array, 
        y: Array,
):
    eps = jnp.linspace(0, 1, 20)
    results = dict()
    for epsilon in eps:
        _, is_adv = FGSM(model, X, y, epsilon)
        results[f"{epsilon:.4f}"] = np.sum(is_adv)
    return results
    

@eqx.filter_jit
def loss(
        model: PyTree,
        X: Array,
        y: Array 
):
    y_pred = jax.vmap(model)(X)
    y_pred = jax.nn.log_softmax(y_pred)
    loss_value = cross_entropy(y_pred, y)
    acc = accuracy(y_pred, y)
    return jnp.mean(loss_value), acc


def train(
        model: PyTree,
        trainloader: DataLoader,
        testloader: DataLoader,
        epochs: Int,
        lr: Float
):
    optim = optax.sgd(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def make_step(model, X, y, opt_state):
        (train_loss, train_acc), grads = eqx.filter_value_and_grad(loss, has_aux=True)(model, X, y)
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, train_loss, train_acc, opt_state

    pbar = tqdm.tqdm(range(epochs))
    train_losses = []
    train_acces = []
    test_losses = []
    test_acces = []
    results = {}

    for iter in pbar:
        for (X, y) in trainloader:
            X = X.numpy()
            y = y.numpy()
            model, train_loss, train_acc, opt_state = make_step(model, X, y, opt_state)
            train_losses.append(train_loss.item())
            train_acces.append(train_acc.item())
        for (X, y) in testloader:
            X = X.numpy()
            y = y.numpy()
            test_loss, test_acc = loss(model, X, y)
            test_losses.append(test_loss.item())
            test_acces.append(test_acc.item())

            res = find_distance(model, X, y)
            res["train_loss"] = train_loss.item()
            res["test_loss"] = test_loss.item()

            results[f"{iter}"] = res
        
        if (iter + 1) % 5 == 0:
            
            train_loss = np.mean(train_losses)
            train_acc = np.mean(train_acces)
            test_loss = np.mean(test_losses)
            test_acc = np.mean(test_acces)

            pbar.set_postfix({
                "Iter" : f"{iter + 1}",
                "train loss" : f'{train_loss:.4f}',
                "train acc" : f'{train_acc:.4f}',
                "val loss" : f'{test_loss:.4f}',
                "val acc" : f'{test_acc:.4f}'
            })

            train_losses = []
            train_acces = []
            test_losses = []
            test_acces = []
    return results


def main():
    key, subkey = jax.random.split(KEY)
    model = eqx.nn.MLP(
        in_size=2,
        out_size=2,
        width_size=128,
        depth=10,
        key=key,
        activation=jax.nn.elu
    )

    X, y = make_data(n_samples=1000, noise=.1, type="circles", SEED=SEED)

    train_idx = jax.random.permutation(subkey, int(y.shape[0] * .8))
    traindata = TensorDataset(tensor(X[train_idx]), tensor(y[train_idx]))
    testdata = TensorDataset(tensor(X[~train_idx]), tensor(y[~train_idx]))

    trainloader = DataLoader(traindata, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(testdata, batch_size=BATCH_SIZE, shuffle=False)

    results = train(
        model,
        trainloader,
        testloader,
        epochs=EPOCHS,
        lr=1e-3
    )

    pd.DataFrame(results).to_parquet("data/nnData.parquet")



if __name__ == "__main__":
    main()