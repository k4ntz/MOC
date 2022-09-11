# SPACE-Time

:exclamation: :boom: Due to VPN issues I could not prepare the files & test the instructions mentioned below. So if anything is not working yet, let me know or maybe fix it directly. :boom: :exclamation:

**Dataset Creation**

We heavily modified the create_dataset.py described below for SPACE-Time:

It tries to use an Agent from folder `agents` also in root directory'
`python3 create_dataset.py -f train -g Pong` would create a dataset based on the trail images

Important new options are:

* `--plot_bb` adds bounding boxes to the generated images! (helps debug)
* `--root` use the global mode image instead of the mode of the trail
* `--root-median` computes the mode and median images from all of those already generated.

If one wants to create a SpaceTime dataset, use:
`python3 create_dataset.py -f train -g Pong` for 100 e.g images
then
`python3 create_dataset.py -f train -g Pong --rootmedian` > generate root.png
Check if the median is proper or redo with more images.
`python3 create_dataset.py -f train -g Pong --root`.

Some other frequently changed things are currently not settings yet, but set in the code...
One such thing is the visualization; in `motion_processing.py` we define the class ProcessingVisualization
and implementations, that take in the frames and motion and visualize sth. for every `self.every_n` up to `self.max_vis`
times.

The settings for the visualization are defined (grouped by motion type and one for the ground truth labels)
inside the main function.
E.g `ZWhereZPres` does show most of the steps used when going from binary motion yes/no to z_where and z_pres latents.

**Loading the model**

The model SPACE-Time is loaded with `Checkpointer` in `src/utils.py`, while the config in `src/configs`
(e.g. `atari_mspacman.yaml`) control which model is loaded. Usage can be seen in `train.py`:

```python
model = get_model(cfg)
model = model.to(cfg.device)

checkpointer = Checkpointer(osp.join(cfg.checkpointdir, cfg.exp_name), max_num=cfg.train.max_ckpt,
                            load_time_consistency=cfg.load_time_consistency, add_flow=cfg.add_flow)
model.train()

optimizer_fg, optimizer_bg = get_optimizers(cfg, model)

if cfg.resume:
    checkpoint = checkpointer.load_last(cfg.resume_ckpt, model, optimizer_fg, optimizer_bg, cfg.device)
    if checkpoint:
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step'] + 1
if cfg.parallel:
    model = nn.DataParallel(model, device_ids=cfg.device_ids)
```

For Space-Time the flags for `load_time_consistency` and `add_flow` are set to `True`, so the model loads properly, but that should be already present in the respective config. Also `resume`should be set when loading a model.

**Usage for downstream-task**

`model.space` then retrieves the included `Space` model. `Space` exposes a method `scene_description` for retrieving the knowledge of a scene:

```python
space = model.space
# x is the image on device as a Tensor, z_classifier accepts the latents,
# only_z_what control if only the z_what latent should be used (see docstring)
scene = space.scene_description(x, z_classifier=z_classifier, only_z_what=True)
# scene now contains a dict[int (the label) -> list[(int, int)]
# (the positions of the object in the domain (-1, 1) that can be mapped into (0, H)
# by using zwhere_to_box (see Appendix in paper).
#
# The label index can be converted into a string label using the index
# of `label_list_*` that are defined in `src/dataset/labels.py`.
```

During evaluation a new only_z_what-style-classifier using `RidgeRegression` is trained with validation data in a few-shot setting and saved along the checkpoints as `z_what-classifier_{samples_per_class}.joblib.pkl`. Here `samples_per_class are the amount of samples from each label class the classifier is trained with.

It can be loaded as:

```python
joblib.load(filename)
```

**Training the model**

Commonly I train the model using:

`python main.py --task multi_train --config configs/atari_riverraid.yaml`

`multi_train` refers to `src/engine/multi_train.py` that enables running multiple trainings/experiments sequentially.
There (following the many commented-out examples) configs that should be altered (relative to the `.yaml`) can be noted.
The script will then iterate over the powerset of configuration options, i.e. every combination of setting. Consider the following configuration as an example:

```python
[
    {
        'seed': ('seed', range(5), identity_print),
        'aow': ('arch.area_object_weight', np.insert(np.logspace(1, 1, 1), 0, 0.0), lambda v: f'{v:.1f}'),
    },
]
```

Here we iterate over five seeds respectively once for `arch.area_object_weight = 0.0` and `arch.area_object_weight = 10.0`. This corresponds to the setting used for evaluation of SPACE-Time and Space-Time without Object Consistency (SPACE-Flow). The third element in the tuple is function to map a compact descriptive string as its used for the experiment name used in the output folder.

Evaluation is run alongside training if the config `train.eval_on` is set to True.

For running the baseline SPACE with the current evaluation and dataset framework please switch to branch `space_upstream`,
where these usage instructions similarly apply.

**AIML Lab specific notes**

The models and datasets will be stored in `/storage-01/ml-trothenbacher/space-time/`, but are currently still stored in my user folder `~/SPACE` distributed over DGX-B, DGX-C and DGX-D as I did not know of the storage option and loading from storage was still really slow at the time. Some configs and steps might not translate directly to the new file structure.


## Loading the model for RL task
There is an example `load_model.py` that allows to load and give a look at the SPACE-time model easily. You can launch the following command from the `src` folder:
`python3 load_model.py --config configs/atari_pong_classifier.yaml`

# SPACE

This is the AIML Rework of the SPACE model presented in the following paper:

> [SPACE: Unsupervised Object-Oriented Scene Representation via Spatial Attention and Decomposition](https://arxiv.org/abs/2001.02407)  

![spaceinv_with_bbox](figures/spaceinvaders.png)

[link to the original repo](https://github.com/zhixuan-lin/SPACE)

## General

Project directories:

* `src`: source code
* `data`: where you should put the datasets
* `output`: anything the program outputs will be saved here. These include
  * `output/checkpoints`: training checkpoints. Also, model weights with the best performance will be saved here
  * `output/logs`: tensorboard event files
  * `output/eval`: quantitative evaluation results
  * `output/demo`: demo images
* `scripts`: some useful scripts for downloading things and showing demos
* `pretrained`: where to put downloaded pretrained models

This project uses [YACS](https://github.com/rbgirshick/yacs) for managing experiment configurations. Configurations are specified with YAML files. These files are in `src/configs`. We provide five YAML files that correspond to the figures in the paper:

* `3d_room_large.yaml`: for the 3D Room Large dataset
* `3d_room_small.yaml`: for 3D Room Small dataset
* `atari_spaceinvaders.yaml`: for the Space Invaders game
* `atari_riverraid.yaml`: for the River Raid game
* `atari_joint.yaml`: for joint training on 10 Atari games

## Dependencies

:bangbang: I made it work with PyTorch 1.8.0 (last version)

If you can use the default CUDA (>=10.2) version, then just use
```
pip3 install -U pip
pip3 install -r requirements.txt
```

## Quick demo with pretrained models

To download pretrained models, two options are available:

* **Download with scripts**. Run the following script to download pretrained models:

  ```
  sh scripts/download_data_atari.sh  # for atari data only
  ```
or
  ```
  sh scripts/download_pretrained.sh  # for all data
  ```

  Pretrained models will be downloaded to the `pretrained` directory and decompressed.

To generate the image of the atari game with the bounding box:
```
sh scripts/show_atari_spaceinvaders.sh 'cuda:0'  # if you have a GPU
sh scripts/show_atari_spaceinvaders.sh 'cpu'  # otherwise
```

## :space_invader: AIML SCRIPTS :space_invader:

We have our own scripts:
* To create a dataset for the game pong (`train` folder) from root folder: <br/>
`python3 create_dataset.py -f train -g Pong`

* To create a dataset for the game Tennis (`train` folder) and make the data i.i.d.: <br/>
`python3 create_dataset.py -f train -g Tennis --random`

All of the following is also called during eval stage when training. Check out train.py and the src/configs to configure.

* To extract images for a game from src folder: <br/>
`python3 post_eval/extract_bb.py --config configs/atari_spaceinvaders.yaml resume True resume_ckpt ../pretrained/atari_spaceinvaders.pth device cuda:0 `

* To create a PCA (or tsne) and visualize in a plot from src: (only available for MsPacman yet) also always check the parameter descriptions of argparse <br/>
`python3 post_eval/classify_z_what.py`

## Training and Evaluation

**First, `cd src`.  Make sure you are in the `src` directory for all commands in this section. All paths referred to are also relative to `src`**.

The general command to run the program is (assuming you are in the `src` directory)

```
python main.py --task [TASK] --config [PATH TO CONFIG FILE] [OTHER OPTIONS TO OVERWRITE DEFAULT YACS CONFIG...]
```

Detailed instructions will be given below.

**Training**. Run one or more of the following to train the model on the datasets you want:

* River Raid:

  ```
  python main.py --task train --config configs/atari_riverraid.yaml resume True device 'cuda:0'
  ```

* Space Invaders:

  ```
  python main.py --task train --config configs/atari_spaceinvaders.yaml resume True device 'cuda:0'
  ```

* Joint training on 10 Atari games:

  ```
  python main.py --task train --config configs/atari_joint.yaml resume True device 'cuda:0'
  ```

These start training with GPU 0 (`cuda:0`). There some useful options that you can specify. For example, if you want to use GPU 5, 6, 7, and 8 and resume from checkpoint `../output/checkpoints/3d_room_large/model_000008001.pth`, you can run the following:

```
python main.py --task train --config configs/3d_room_large.yaml \
	resume True resume_ckpt '../output/checkpoints/3d_room_large/model_000008001.pth' \
	parallel True device 'cuda:5' device_ids '[5, 6, 7, 8]'
```

Other available options are specified in `config.py`.

**Training visualization**. Run the following

```
# Run this from the 'src' directory
tensorboard --bind_all --logdir '../output/logs' --port 8848
```

And visit `http://[your server's address]:8848` in your local browser.

## Issues

* For some reason we were using BGR images for our Atari dataset and our pretrained models can only handle that. Please convert the images to BGR if you are to test your own Atari images with the provided pretrained models.
* There is a chance that SPACE doesn't learn proper background segmentation for the 3D Room Large datasets. Due to the known [PyTorch reproducibity issue](https://pytorch.org/docs/stable/notes/randomness.html), we cannot guarantee each training run will produce exactly the same result even with the same seed. For the 3D Room Large datasets, if the model doesn't seem to be segmenting the background in 10k-15k steps, you may considering changing the seed and rerun (or not even changing the seed, it will be different anyway). Typically after trying 1 or 2 runs you will get a working version.

## Use SPACE for other tasks

If you want to apply SPACE to your own task (e.g., for RL), please be careful. Applying SPACE to RL is also our original intent, but we found that the model can sometimes be unstable and sensitive to hyperparameters and training tricks. There are several reasons:

1. **The definition of objects and background is ambiguous in many cases**. Atari is one case where objects are often well-defined. But in many other cases, it is not. For more complicated datasets, making SPACE separate foreground and background properly can be something non-trivial.
2. **Learning is difficult when object sizes vary a lot**. In SPACE, we need to set a proper prior for object sizes manually and that turn out to be crucial hyperparameter. For example, for the 10 Atari games we tested, objects are small and roughly of the same size. When object sizes vary a lot SPACE may fail.

That said, we are pleased to offer discussions and pointers if you need help (especially when fine-tuning it on your own dataset). We also hope this will facilitate future works that overcome these limitations.


## Acknowledgements

Please refer to [the original model](https://github.com/zhixuan-lin/SPACE) for this.
