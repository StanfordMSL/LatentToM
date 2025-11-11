# Latent Theory of Mind: A Decentralized Diffusion Architecture for Cooperative Manipulation
This repository is the official implementation of [Latent Theory of Mind: A Decentralized Diffusion Architecture for Cooperative Manipulation](https://marmotlab.org/publications/86-CORL2025-LatentToM.pdf). 
The paper is accepted for Oral Presentation at the Conference on Robot Learning (CoRL 2025).
This code implementation is based on the official [Diffusion Policy repository](https://github.com/real-stanford/diffusion_policy) and can be easily integrated into the official repository as a branch.

![alt](overview.png)

## Environment Configuration
We use the same environment provided by the official [Diffusion Policy repository](https://github.com/real-stanford/diffusion_policy).
```
conda env create -f conda_environment_real.yaml
```

## Start Training
Once we have confirmed that the environment has been configured, we can activate it with the following command:
```
conda activate robodiff
```
Then, we can run the main program to start training:
```
python train.py --config-name=sheaf_xarm_split_diffusion_workspace
```
The above is the same implementation we provided as in the main text, meaning that the two arms will have shared third-persion view. If you want each arm to have its own separate third-view camera, you can use the following command:
```
python train.py --config-name=sheaf_individual_camera_diffusion_workspace
```

## Task Results
We have two experiments: cooperative push-T and coffee bean pouring. We are only providing the training data for coffee bean pouring here.

![alt](result_task1_v2.png)
![alt](result_task2.png)

## Reference
If this repository is helpful to you, please cite our work by:
```
@article{he2025latent,
  title={Latent Theory of Mind: A Decentralized Diffusion Architecture for Cooperative Manipulation},
  author={He, Chengyang and Camps, Gadiel Sznaier and Liu, Xu and Schwager, Mac and Sartoretti, Guillaume},
  journal={arXiv preprint arXiv:2505.09144},
  year={2025}
}
```