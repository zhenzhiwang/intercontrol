# InterControl
This repository is the official implementation of [InterControl](http://arxiv.org/abs/2311.15864).

**[InterControl : Generate Human Motion Interactions by Controlling Every Joint](http://arxiv.org/abs/2311.15864)**

**Zhenzhi Wang**, Jingbo Wang, Dahua Lin, Bo Dai.

## Interaction Demo
Due to HumanML3D has no hand joints, interactions involving hands are achieved by setting distances between wrists. It is the reason of penetrations of hands. Captions are explanations for joint contact pairs, not text prompts.
<table class="center">
 <tr style="line-height: 0">
  <td style="border: none; text-align: center">A person uses right foot to kick another's left shoulder.</td>
  <td style="border: none; text-align: center">Two people's right feet are contacted while left feet are seperated by 1.8m.</td>
  <td style="border: none; text-align: center">Three people's right feet are contacted together.</td>
  </tr>
    <tr>
    <td><img src="./assets/kicks_shoulder.gif"></td>
    <td><img src="./assets/foot_pull_and_push.gif"></td>
    <td><img src="./assets/three_people_foot.gif"></td>
    </tr>
</table>
<p style="margin-left: 2em; margin-top: -1em"></p>


<table class="center">
<tr style="line-height: 0">
  <td style="border: none; text-align: center">Person A uses right wrist to touch Person B's right shoulder, then Person B uses left wrist to touch Person C's left shoulder.</td>
  <td style="border: none; text-align: center">Two people's right wrists are contacted while left wrists are seperated by 2.4m.</td>
  <td style="border: none; text-align: center">Two people are shaking hands with right wrists.</td>
  </tr>
    <tr>
    <td><img src="./assets/three_people.gif"></td>
    <td><img src="./assets/hand_push_and_pull.gif"></td>
    <td><img src="./assets/shake_hands.gif"></td>
    </tr>
</table>
<p style="margin-left: 2em; margin-top: -1em"></p>


<table class="center">
    <tr style="line-height: 0">
  <td style="border: none; text-align: center">Two people uses right feet to kick each other.</td>
  <td style="border: none; text-align: center">A person chases another person and shakes hands by right wrists.</td>
  <td style="border: none; text-align: center">A person chases another person and put left/right hands on another one's left/right shoulder.</td>
  </tr>
    <tr>
    <td><img src="./assets/foot_kicks_foot.gif"></td>
    <td><img src="./assets/back_shake_hands.gif"></td>
    <td><img src="./assets/put_hand_on_shoulder_from_back.gif"></td>
    </tr>
</table>
<p style="margin-left: 2em; margin-top: -1em"></p>


## Abstract
Text-conditioned human motion generation model has achieved great progress by introducing diffusion models and corresponding control signals. However, the interaction between humans are still under explored. To model interactions of arbitrary number of humans, we define interactions as human joint pairs that are either in contact or separated, and leverage Large Language Model (LLM) Planner to translate interaction descriptions into contact plans. Based on the contact plans, interaction generation could be achieved by spatially controllable motion generation methods by taking joint contacts as spatial conditions. We present a novel approach named InterControl for flexible spatial control of every joint in every person at any time by leveraging motion diffusion model only trained on single-person data. We incorporate a motion controlnet to generate coherent and realistic motions given sparse spatial control signals and a loss guidance module to precisely align any joint to the desired position in a classifier guidance manner via Inverse Kinematics (IK). Extensive experiments on HumanML3D and KIT-ML dataset demonstrate its effectiveness in versatile joint control. We also collect data of joint contact pairs by LLMs to show InterControl's ability in human interaction generation.

## Getting started

Our code is developed from [PriorMDM](https://arxiv.org/abs/2303.01418), therefore shares similar dependencies and setup instructions, which requires:

* Python 3.8
* conda3 or miniconda3
* CUDA capable GPU (one is enough)

### 1. Setup environment following PriorMDM instructions

Install ffmpeg (if not already installed):

```shell
sudo apt update
sudo apt install ffmpeg
```
For windows use [this](https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/) instead.

Setup conda env:
```shell
conda env create -f environment.yml
conda activate PriorMDM
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/GuyTevet/smplx.git
```

### 2. Get MDM dependencies

<details>
  <summary><b>If you already have an installed MDM</b></summary>

**Link from installed MDM**

Before running the following bash script, first change the path to the full path to your installed MDM

```bash
bash prepare/link_mdm.sh
```

</details>


<details>
  <summary><b>First time user</b></summary>

**Download dependencies:**

```bash
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
```

**Get HumanML3D dataset** :

Follow the instructions in [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git),
then copy the result dataset to our repository:

```shell
cp -r ../HumanML3D/HumanML3D ./dataset/HumanML3D
```

</details>

### 3. Download MDM pretrained models

Download the model(s) you wish to use, then unzip and place it in `./save/`.

* [my_humanml-encoder-512](https://drive.google.com/file/d/1RCqyKfj7TLSp6VzwrKa84ldEaXmVma1a/view?usp=share_link) (This is a reproduction of MDM best model without any changes)


## Single-Person Motion Generation

### 1. Training
The model will save in the directory `./save/`.

Loss Guidance on $\mu_t$
```shell
python3 -m train.train_global_joint_control --save_dir save/posterior_mask1_bfgs5_humanml_all \
--dataset humanml --inpainting_mask global_joint --lr 0.00001 --mask_ratio 1 --control_joint all \
--use_posterior --bfgs_times_first 5
```

Loss Guidance on $x_0$

```shell
python3 -m train.train_global_joint_control --save_dir save/mask1_bfgs0_humanml_all \
--dataset humanml --inpainting_mask global_joint --lr 0.00001 --mask_ratio 1 --control_joint all \
--bfgs_times_first 0
```
### 2. Evaluation 
Select checkpoint to be evluated by sepcifying the `model_path`, and use `replication_times` for multiple evaluations and get average results, the following evaluation script will generate motions for 10 times.

Loss Guidance on $\mu_t$
```shell
python3 -m eval.eval_controlmdm --model_path save/posterior_mask1_bfgs5_humanml_all/model000120000.pt \
--replication_times 10 --mask_ratio 1 --bfgs_times_first 5 \
--bfgs_times_last 10 --bfgs_interval 1 --use_posterior \
--control_joint all 
```

Loss Guidance on $x_0$

```shell
python3 -m eval.eval_controlmdm --model_path save/mask1_bfgs1_humanml_all/model000120000.pt \
--replication_times 10 --mask_ratio 1 --bfgs_times_first 1 \
--bfgs_times_last 10 --bfgs_interval 1 \
--control_joint all 
```

### 3. Sampling
Loss Guidance on $\mu_t$
```shell
python -m sample.global_joint_control --model_path save/mask1_bfgs1_humanml_all/model000120000.pt \
--num_samples 32 --use_posterior --control_joint all
```

## Human Interaction Generation
### 1. Evaluation 
Loss Guidance on $\mu_t$
```shell
python3 -m eval.eval_interaction --model_path save/posterior_mask0.25_bfgs5_humanml_all/model000120000.pt \
--replication_times 10 --bfgs_times_first 5 --bfgs_times_last 10 --bfgs_interval 1 \
--use_posterior  --control_joint all \
--interaction_json './assets/all_plans.json' \
--multi_person
```
### 2. Sampling
Two-people Interaction Sampling
```shell
python -m sample.interactive_global_joint_control \
--model_path save/posterior_mask0.25_bfgs5_humanml_all/model000120000.pt \
--multi_person --bfgs_times_first 5 --bfgs_times_last 10 \
--interaction_json './assets/sample.json' \
```

More than 3 people interaction sampling, need hand-crafted masks for each person
```shell
python -m sample.more_people_global_joint_control \
--model_path save/posterior_mask0.25_bfgs5_humanml_all/model000120000.pt \
--multi_person --bfgs_times_first 5 --bfgs_times_last 10 --use_posterior \
```
## Acknowledgments

This code is standing on the shoulders of giants. We want to thank the following contributors
that our code is based on:

[GMD](https://github.com/korrawe/guided-motion-diffusion),
[PriorMDM](https://github.com/priorMDM/priorMDM),
[MDM](https://github.com/GuyTevet/motion-diffusion-model),
[guided-diffusion](https://github.com/openai/guided-diffusion), 
[MotionCLIP](https://github.com/GuyTevet/MotionCLIP), 
[text-to-motion](https://github.com/EricGuo5513/text-to-motion), 
[actor](https://github.com/Mathux/ACTOR), 
[joints2smpl](https://github.com/wangsen1312/joints2smpl),
[TEACH](https://github.com/athn-nik/teach).

## License
This code is distributed under an [MIT LICENSE](LICENSE).

Note that our code depends on other libraries, including CLIP, SMPL, SMPL-X, PyTorch3D, and uses datasets that each have their own respective licenses that must also be followed.
