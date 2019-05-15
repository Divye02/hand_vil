# visuomotor-hand-man
Learning Deep Visuomotor Policies for Dexterous Hand Manipulation

## Background

The overall project is built on top of these three repositories:

1. [mjrl](https://github.com/aravindr93/mjrl) provides a suite of learning algorithms for various continuous control tasks simulated in MuJoCo. This includes the NPG implementation and the DAPG algorithm used in the paper.
2. [mj_envs](https://github.com/vikashplus/mj_envs) provides a suite of continuous control tasks simulated in MuJoCo, including the dexterous hand manipulation tasks used in the paper.
3. [hand_dapg](https://github.com/aravindr93/hand_dapg) (this repository) serves as the landing page and contains the human demonstrations and pre-trained policies for the tasks.

## Setup

Each repository above contains detailed setup instructions. 
1. **Step 1:** Install [mjrl](https://github.com/aravindr93/mjrl), using instructions in the repository ([direct link](https://github.com/aravindr93/mjrl/tree/master/setup)). `mjrl` comes with an anaconda environment which helps to easily import and use a variety of MuJoCo tasks.
2. **Step 2:** Install [mj_envs](https://github.com/vikashplus/mj_envs) by following the instructions in the repository. Note that `mj_envs` uses git submodules, and hence must be cloned correctly per instructions in the repo.
3. **Step 3:** After setting up `mjrl` and `mj_envs`, Add them to your python path.
```
$ export PYTHONPATH=$PYTHONPATH:<your_path>/mjrl
$ export PYTHONPATH=$PYTHONPATH:<your_path>/mj_envs
```

## Training the Visuomotor policies

1. **Step 1:** Clone this repo. Replace the environment XML files in mj_envs with that present in mjrl_mod/envs/assests, as we need to add the cameras for training the visuomotor policies. Namely:
* mj_envs/hand_manipulation_suite/assets/DAPG_door.xml to be relaced by mjrl_mod/envs/assets/DAPG_door.xml
* mj_envs/hand_manipulation_suite/assets/DAPG_hammer.xml to be relaced by mjrl_mod/envs/assets/DAPG_hammer.xml
* mj_envs/hand_manipulation_suite/assets/DAPG_pen.xml to be relaced by mjrl_mod/envs/assets/DAPG_pen.xml
* mj_envs/hand_manipulation_suite/assets/DAPG_relocate.xml to be relaced by mjrl_mod/envs/assets/DAPG_relocate.xml

2. **Step 2** We already have the expert policies for each of the environments fetched from [hand_dapg](https://github.com/aravindr93/hand_dapg). So we are ready to train the visual policy for any of the above 4 environments.
* It is highly reccomended that you use a machine with a GPU for faster training.
* All the training for the different environments are present in configs/
* ~~Step about copying the config and placing it in the correct way>~~
* Now, we are ready the train the visual model.
```
$ python run.py
```
 
## Bibliography

If you use the code in this or associated repositories above, please cite the following paper.
```
~~Put Cutation Here~~
```
