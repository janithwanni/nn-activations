import torch
from data_utils import *
from model_utils import *
from neuron_plot import *
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import secrets
import pandas as pd
import time
import gc
# ===== 

# df = pd.read_csv("df.csv")
# ======

def execute(neuron, seed, train, test):
    if neuron < 5:
        epoch = 500
    else: 
        epoch = 500 # 100 maybe let's keep it fair 
    # fixed data
    # start model with different seed
    layer_sizes = [neuron]

    data = train.loc[:, ["x", "y"]].values
    y = np.array([1.0 if v == "A" else 0.0 for v in train.loc[:, "class"].values.tolist()])
    model = run_model(
        data,
        y,
        epochs=epoch,
        layer_sizes = layer_sizes,
        torch_seed=seed
    )
    
    evals = evaluate(model, test)
    del train
    del test
    del model 
    gc.collect()
    # evals = {"f":1,"acc":2}
    return {"neuron": neuron, "epoch": epoch, "seed": seed, "f1": evals["f"], "acc":evals["acc"]}

def _execute(args):
    print(args)
    train_df = pd.read_csv("train_df.csv")
    test_df = pd.read_csv("test_df.csv")
    e = execute(args[0], args[1], train_df, test_df)
    del train_df
    del test_df
    gc.collect()
    return e

def process_tasks(combinations, max_tasks, max_workers):
    results = []
    total_tasks = len(combinations)

    for i in range(0, total_tasks, max_tasks):
        print(f"Starting new ProcessPoolExecutor for tasks {i} to {i + max_tasks}")

        with ProcessPoolExecutor(max_workers=max_workers, initializer=worker_init) as executor:
            futures = [executor.submit(_execute, combinations[j]) for j in range(i, min(i + max_tasks, total_tasks))]
            for future in as_completed(futures):
                results.append(future.result())  # Process each result immediately

        print(f"Finished batch {i} to {i + max_tasks}, restarting executor...\n")
    return results

def worker_init():
    gc.collect()
# ===== 

if __name__ == "__main__":
    # Define possible values for each argument
    neuron_values = ([4] * 50) + ([5] * 50)
    # neuron_values = ([5] * 50)
    seeds_to_try = [secrets.randbelow(99999) for i in range(100)]
    # Generate all possible combinations
    combinations = list(zip(neuron_values, seeds_to_try))
    
    #with ProcessPoolExecutor(max_workers = 2, initializer=worker_init) as executor:
        # futures = [executor.submit(_execute, combinations[i]) for i in range(len(combinations))]

        # results = [future.result() for future in as_completed(futures)]
        # results = list(executor.map(_execute, combinations))
    results = process_tasks(combinations, max_tasks = 5, max_workers = 4)
    res_df = pd.DataFrame.from_dict(results)
    res_df.to_csv(f"res_df_{int(time.time())}.csv", index=False)
