# RL-for-FJSSP

This repository contains the implementation of the paper [Diverse policy generation for the flexible job-shop scheduling problem via deep reinforcement learning with a novel graph representation]([https://arxiv.org/abs/2310.15706](https://www.sciencedirect.com/science/article/pii/S0952197624016464))

## Get started

### Summary

In the ```data``` folder the generated test set and the benchmarks can be found. In the ```models``` folders the final result is saved. The ```src``` folder contains:

* ```cluster_policies.py``` to cluster the genearted candidate policies.
* ```env.py``` the FJSSP as a MDP.
* ```generate_val.py``` to generate the validation set.
* ```generator.py``` to generate random instances.
* ```parsedata.py``` to parse FJSSP instance.
* ```ppo.py``` our adapted PPO implementation with Graph Neural Networks.
* ```solver.py``` the OR-Tools solver to solve FJSSP instances.
* ```train.py``` the functions to train a network.


### Dependecies

To install dependencies run

```
pip install -r requirements.txt
```

This code was developped using Python 3.10, but lower versions should work.

### Train

To generate a set of scheduling policies run

```
python main.py
```

### Test

To test the policies add the intances to the data folder and run 

```
python test.py
```

## Citation

If you find our work useful, please consider citing:

```
@article{echeverria2023solving,
  title={Solving large flexible job shop scheduling instances by generating a diverse set of scheduling policies with deep reinforcement learning},
  author={Echeverria, Imanol and Murua, Maialen and Santana, Roberto},
  journal={arXiv preprint arXiv:2310.15706},
  year={2023}
}
```
