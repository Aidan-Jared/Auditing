from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import f1_score, accuracy_score

from art.attacks.evasion import DecisionTreeAttack
from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeClassifier

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import numpy as np

import json

from collections import deque

SEED = 42
key = jax.random.PRNGKey(SEED)

def make_data(n_samples, noise, type = "moons", SEED=42):
    if type == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=SEED, shuffle=True)
    elif type == "circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, random_state=SEED, shuffle=True)
    else:
        key = jax.random.PRNGKey(SEED)
        key1, key2 = jax.random.split(key)
        X = jax.random.uniform(key1, (n_samples * 4, 2), minval=-3, maxval=3, dtype=jnp.float32)
        y = ((jnp.floor(X[:,0]) + jnp.floor(X[:,1])) % 2).astype(int)
        noise_idx = jax.random.choice(key2, a= n_samples*4, shape = (int(n_samples * 4 * noise),), replace=False)
        y = y.at[noise_idx].set(1 - y[noise_idx])
    return X, y


def audit_model(
        X,
        y, 
        SEED : int =42, 
        max_depths : list =[1], 
        max_features :int = 2,
        stopping = 5
        ):
    acc_queue = deque(maxlen=stopping)
    acc_queue.extend([0,.1])
    d = 1
    results = dict()
    # for d in max_depths:

    while np.std(acc_queue).item() >=.001:
        model = DecisionTreeClassifier(
            random_state=SEED, max_features=max_features,
            max_depth=d, min_samples_split=2, min_samples_leaf=1)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        train_acces = []
        train_f1s = []
        test_acces = []
        test_f1s = []
        adv_acces = []
        adv_f1s = []
        adv_distances = []


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
            x_attack_adv = attack.generate(x=X_test)

            
            y_adv_pred, acc_adv, f1_adv = eval_model(x_attack_adv, y_test.reshape(y_test.shape[0],1), classifier)
            adv_distance = np.linalg.norm(X_test - x_attack_adv)
            train_acces.append(train_acc)
            train_f1s.append(train_f1)
            test_acces.append(test_acc)
            test_f1s.append(test_f1)
            adv_acces.append(acc_adv)
            adv_f1s.append(f1_adv)
            adv_distances.append(adv_distance)

        acc_queue.append(np.mean(test_acces).item())

        res = {"train acc": np.mean(train_acces).item(), 
               "train f1" : np.mean(train_f1s).item(), 
               "test acc": np.mean(test_acces).item(), 
               "test f1" : np.mean(test_f1s).item(), 
               "adv acc": np.mean(adv_acces).item(), 
               "adv f1": np.mean(adv_f1s).item(), 
               "adv distance" : np.mean(adv_distances).item()}
        results[f"max_depth = {d}"] = res
        d += 1
        
    return results


                
def eval_model(X, y, classifier):
    y_pred = classifier.predict(X)
    acc = accuracy_score(y, np.argmax(y_pred, axis = 1))
    f1 = f1_score(y, np.argmax(y_pred, axis = 1))
    return y_pred, acc, f1

def main():
    X, y = make_data(1000, .2, "moons")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
    plt.show()
    results = audit_model(X, y, max_depths=[1,3,5,7,9,10,11,12,13,15,20])
    
    with open("data/data.json", 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
