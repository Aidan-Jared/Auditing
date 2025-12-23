from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_circles, make_moons, make_blobs
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import f1_score, accuracy_score

from art.attacks.evasion import DecisionTreeAttack, FastGradientMethod
from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeClassifier, ScikitlearnSVC

import polars as pl

import jax
import jax.numpy as jnp
from jaxtyping import PyTree,  Array, Float
import equinox as eqx

import numpy as np

from collections import deque


@eqx.filter_jit
def FGSM(
        model: PyTree,
        x: Array,
        y: Array,
        epsilon: Float
) ->tuple[Array, Array]:

    def loss_fn(x):
        y_pred = jax.vmap(model)(x)
        one_hot = jax.nn.one_hot(y, num_classes=2)
        loss = -jnp.sum(one_hot * jax.nn.log_softmax(y_pred))
        return loss

    grad = eqx.filter_grad(loss_fn)(x)

    x_adv = x + epsilon * jnp.sign(grad)

    pred_clean = jnp.argmax(jax.vmap(model)(x), axis=1)

    pred_adv = jnp.argmax(jax.vmap(model)(x_adv), axis=1)
    is_adv = pred_clean != pred_adv
    
    return x_adv, is_adv

def make_data(n_samples, noise, type = "moons", SEED=42):
    if type == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=SEED, shuffle=True)
        # X, y = make_blobs(n_samples=n_samples, random_state=SEED, n_features=2, centers=2)
    elif type == "circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, random_state=SEED, shuffle=True)
    else:
        key = jax.random.PRNGKey(SEED)
        key1, key2 = jax.random.split(key)
        X = jax.random.uniform(key1, (n_samples, 2), minval=-3, maxval=3, dtype=jnp.float32)
        y = ((jnp.floor(X[:,0]) + jnp.floor(X[:,1])) % 2).astype(int)
        noise_idx = jax.random.choice(key2, a= n_samples, shape = (int(n_samples * noise),), replace=False)
        y = y.at[noise_idx].set(1 - y[noise_idx])
        X = np.array(X)
        y = np.array(y)
    return X, y


def audit_tree(
        X,
        y, 
        SEED : int =42, 
        max_features :int = 2,
        stopping : int = 5,
        n_splits : int = 5,
        top_k : int = 5
        ):
    acc_queue = deque(maxlen=stopping)
    acc_queue.extend([0,.1])
    d = 1
    results = pl.DataFrame()
    # for d in max_depths:

    while np.std(acc_queue).item() >=.001:
        model = DecisionTreeClassifier(
            random_state=SEED, max_features=max_features,
            max_depth=d, min_samples_split=2, min_samples_leaf=1)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        train_acces = []
        train_f1s = []
        test_acces = []
        test_f1s = []
        avg_acces = []
        avg_f1s = []
        avg_distances = []


        for train_idx, test_idx in skf.split(X, y):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]
            model.fit(X_train,y_train)

            classifier = ScikitlearnDecisionTreeClassifier(model)

            y_pred, test_acc, test_f1 = eval_model(X_test, y_test.reshape(y_test.shape[0],1), classifier)
            y_train_pred, train_acc, train_f1 = eval_model(X_train, y_train.reshape(y_train.shape[0],1), classifier)
            

            attack = DecisionTreeAttack(classifier=classifier)
            try:
                x_attack_adv = attack.generate(x=X_test)
            except:
                break

            
            y_adv_pred, acc_adv, f1_adv = eval_model(x_attack_adv, y_test.reshape(y_test.shape[0],1), classifier)
            distances = []
            for idx, attack_data in enumerate(x_attack_adv):
                true_label = y_test[idx]
                attack_class_points = X_test[y_test != true_label]
                distance_vector = np.sqrt(np.sum((attack_data - attack_class_points)**2, axis=1))
                indices = np.argsort(distance_vector)[:top_k]
                knn_distance = distance_vector[indices]
                distances.append(np.mean(knn_distance).item())


            adv_distance = np.mean(distances)
            train_acces.append(train_acc)
            train_f1s.append(train_f1)
            test_acces.append(test_acc)
            test_f1s.append(test_f1)
            avg_acces.append(acc_adv)
            avg_f1s.append(f1_adv)
            avg_distances.append(adv_distance.tolist())

        acc_queue.append(np.mean(test_acces).item())

        res = {"train acc": train_acces, 
               "train f1" : train_f1s, 
               "test acc": test_acces, 
               "test f1" : test_f1s, 
               "adv acc": avg_acces, 
               "adv f1": avg_f1s, 
               "adv distance" : avg_distances,
               "depth" : d
               }
        results = pl.concat([results, pl.from_dict(res)])
        d += 1
        
    return results

def audit_svc(
        X,
        y, 
        SEED : int =42, 
        C: float = .3,
        stopping : int = 5,
        n_splits : int = 5,
        top_k : int = 5
        ):
    acc_queue = deque(maxlen=stopping)
    acc_queue.extend([0,.1])
    d = 1
    results = pl.DataFrame()
    # for d in max_depths:

    while np.std(acc_queue).item() >=.001:
        model = SVC(
            random_state=SEED, C=C, kernel="rbf")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        train_acces = []
        train_f1s = []
        test_acces = []
        test_f1s = []
        avg_acces = []
        avg_f1s = []
        avg_distances = []


        for train_idx, test_idx in skf.split(X, y):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]
            model.fit(X_train,y_train)

            classifier = ScikitlearnSVC(model)

            y_pred, test_acc, test_f1 = eval_model(X_test, y_test.reshape(y_test.shape[0],1), classifier)
            y_train_pred, train_acc, train_f1 = eval_model(X_train, y_train.reshape(y_train.shape[0],1), classifier)
            

            attack = FastGradientMethod(estimator=classifier, eps=.07)
            try:
                x_attack_adv = attack.generate(x=X_test)
            except:
                break

            
            y_adv_pred, acc_adv, f1_adv = eval_model(x_attack_adv, y_test.reshape(y_test.shape[0],1), classifier)
            distances = []
            for idx, attack_data in enumerate(x_attack_adv):
                true_label = y_test[idx]
                attack_class_points = X_test[y_test != true_label]
                distance_vector = np.sqrt(np.sum((attack_data - attack_class_points)**2, axis=1))
                indices = np.argsort(distance_vector)[:top_k]
                knn_distance = distance_vector[indices]
                distances.append(np.mean(knn_distance).item())


            adv_distance = np.mean(distances)
            train_acces.append(train_acc)
            train_f1s.append(train_f1)
            test_acces.append(test_acc)
            test_f1s.append(test_f1)
            avg_acces.append(acc_adv)
            avg_f1s.append(f1_adv)
            avg_distances.append(adv_distance.tolist())

        acc_queue.append(np.mean(test_acces).item())

        res = {"train acc": train_acces, 
               "train f1" : train_f1s, 
               "test acc": test_acces, 
               "test f1" : test_f1s, 
               "adv acc": avg_acces, 
               "adv f1": avg_f1s, 
               "adv distance" : avg_distances,
               "depth" : d
               }
        results = pl.concat([results, pl.from_dict(res)])
        d += 1
        
    return results

def eval_model(X, y, classifier):
    y_pred = classifier.predict(X)
    acc = accuracy_score(y, np.argmax(y_pred, axis = 1))
    f1 = f1_score(y, np.argmax(y_pred, axis = 1))
    return y_pred, acc, f1