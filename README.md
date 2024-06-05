# Self-Supervised Training of Interpretable Neural Networks for Medical Applications

## About
This code is based on my guided research with the [AI Med Lab](https://ai-med.de/) under the supervision of Tom Nuno Wolf at the Technical University of Munich. You can check out my report in `report.pdf` and my presentation in `presentation.pdf`.

We developed two methods called **PCL-ProtoPNet** and **PCL-NW**, which aim to integrate *_interpretability_* of neural networks and *_self-supervised learning_* on unlabelled datasets.

The ”[PCL](https://arxiv.org/abs/2005.04966)” component of the method allows self-supervised learning, while the ”[ProtoPNet](https://arxiv.org/abs/1806.10574)” or ”[NW](https://arxiv.org/abs/2212.03411)” component provides interpretability.

We evaluate our methods on the challenging 3-way AD classification task (AD/MCI/CN) using 3D brain MRI images. We use PCL as a pre-training step on the unlabelled UK BioBank (UKBB) dataset. Subsequently, we either fix or finetune the resulting encoder and classification head on the labelled Alzheimer’s Disease Neuroimaging Initiative (ADNI) dataset.

## Installation
1. Create a conda environment: `conda create -n pcl-protopnet-nw python=3.10`
2. Inside the environment, install all modules in requirements.txt: `pip install -r requirements.txt`
2. Inside the environment, install torchpanic by following the instructions [here](https://github.com/ai-med/PANIC/tree/753847dc8af3027d4946602b11a47142f055e7d8?tab=readme-ov-file#installation).

## Getting Started
1. Configure the paths (see the "Path Configurations" section).
2. Configure the datasets (see the "Dataset Configurations" section).
3. Execute PCL pre-training on UKBB/ADNI/both (see the "PCL Pre-training" section).
4. To train PCL-ProtoPNet, see the "PCL-ProtoPNet" section.
5. To train PCL-NW, see the "PCL-NW" section.

## Path Configurations
In `pcl/paths.py`, you have to set the values for:
- `ADNI_DATA_PATH`: The path to the ADNI dataset which contains HDF5 (.h5) files (we follow the data format described [here](https://github.com/ai-med/PANIC/tree/753847dc8af3027d4946602b11a47142f055e7d8?tab=readme-ov-file#data)).
- `TENSORBOARD_DIR`: The path to the tensorboard directory which contains outputs and logs of experiments.
- `MLFLOW_DIR`: The path to the mlflow directory which contains outputs and logs of experiments.

## Dataset Configurations

### UK Biobank (UKBB)
To use the UKBB dataset, we built:
1. `UkbbMRIDataset`: Used for plain PCL pre-training.
To use this, you first need to set up a CSV file that contains the paths to the individual .nii.gz files. The paths should be listed under the header "path". The path to this CSV file will then be used as an argument to the class. (You can see a sample CSV file in `data_split/img.csv`)
2. `UkbbMRITabDataset`: Used for y-aware PCL pre-training.
To use this, you first need to set up a CSV file that contains the paths to the individual .nii.gz files AND the features associated to each scan. The paths should be listed under the header "path". The path to this CSV file will then be used as an argument to the class. (You can see a sample CSV file in `data_split/img_tab.csv`)

### ADNI
To use the ADNI dataset, we built:
1. `AdniMRIDataset`: Used for plain PCL pre-training.
To use this, you need the path to the .h5 file that contains the ADNI samples.
2. `AdniMRIDataset_nonPCL`: Used for supervised learning.
To use this, you need the path to the .h5 file that contains the ADNI samples.

## PCL Pre-training

You can execute PCL pre-training on the UKBB or ADNI dataset or both. Example:
```
python main_pcl.py --arch [densenet|3dresnet] --dataset [ukbb|adni|ukbb_adni] --workers 8 --batch-size 8 --accumulation-steps 32 --epochs 80 --warmup-epoch 10 --latent_dim 128 --print-freq 1 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --pcl-r 16 --num-cluster 24,96 --exp-dir <pcl_exp_dir> [--use_ukbb_metadata]
```

Note: The flag `use_ukbb_metadata` supports y-aware PCL pre-training using features associated with the scan.

Output:
1. a PCL-pretrained encoder
2. clustering results (centroids, etc). The centroids correspond to unpushed prototypes.

This training step is monitored using tensorboard. The logs and the output are saved under the `<TENSORBOARD_DIR>/<pcl_exp_dir>` directory.

## PCL-ProtoPNet
To build PCL-ProtoPNet:
1. Push the cluster centroids from PCL pre-training to training samples in the ADNI dataset:
    ```
    python push_prototypes_protopnet.py --adni_fold_idx [0|1|2|3|4] --arch [3dresnet|densenet] --pcl_encoder_checkpoint_path <pcl_exp_dir> --latent_dim <pcl_latent_dim>
    ```
    Output: Pushed prototypes corresponding to training samples, saved under the `<TENSORBOARD_DIR>/<pcl_exp_dir>` directory of PCL.
2. You can either fix / train a FC head on top of the PCL-pretrained encoder.

    If you want to fix the head, you can directly run evaluation after PCL pre-training by:
    ```
    python eval_fchead.py --adni_fold_idx [0|1|2|3|4] --adni_eval_data_split [valid|test] --arch [3dresnet|densenet] --encoder_checkpoint_path <pcl_encoder_dir> --latent_dim <pcl_latent_dim> --num_prototypes <pcl_num_prototypes>
    ```
    If you want to train the head, you can run training by:
    ```
    python train_fchead.py --mlflow_exp_name <fchead_exp_name> --arch [3dresnet|densenet] --pcl_encoder_checkpoint_path <pcl_exp_dir> --latent_dim <pcl_latent_dim> --num_prototypes <pcl_num_prototypes> --num_epochs 100 --adni_fold_idx [0|1|2|3|4] --batch_size 8
    ```
    By executing this script, you are conducting a hyperparameter search using [Bayesian optimization](https://github.com/bayesian-optimization/BayesianOptimization). This training step is monitored using mlflow. The logs are saved under the `<MLFLOW_DIR>` directory.

    After training the head, you can run evaluation by:
    ```
    python eval_fchead.py --adni_fold_idx [0|1|2|3|4] --adni_eval_data_split [valid|test] --arch [3dresnet|densenet] --encoder_checkpoint_path <pcl_encoder_dir> --model_checkpoint_path <trained_head_dir> --latent_dim <pcl_latent_dim> --num_prototypes <pcl_num_prototypes> --head_trained
    ```

## PCL-NW
To build PCL-NW, you can either fix / finetune the PCL-pretrained encoder.

If you want to fix the encoder, you can directly run evaluation after PCL pre-training by:
```
python eval_nwhead.py --adni_fold_idx [0|1|2|3|4] --adni_eval_data_split [valid|test] --arch [3dresnet|densenet] --encoder_checkpoint_path <pcl_exp_dir> --latent_dim <pcl_latent_dim>
```

If you want to finetune the encoder together with the NW head, you can run training by:
```
python train_nwhead.py --exp_dir <nwhead_exp_dir> --arch [3dresnet|densenet] --pcl_encoder_checkpoint_path <pcl_exp_dir> --latent_dim <pcl_latent_dim> --num_epochs 200 --adni_fold_idx [0|1|2|3|4] --batch_size 1 --no_data_aug
```
This training step is monitored using tensorboard. The logs are saved under the `<TENSORBOARD_DIR>/<nwhead_exp_dir>` directory.

After finetuning the encoder, you can run evaluation by:
```
python eval_nwhead.py --adni_fold_idx [0|1|2|3|4] --adni_eval_data_split [valid|test] --arch [3dresnet|densenet] --encoder_checkpoint_path <finetuned_encoder_dir> --latent_dim <pcl_latent_dim> --encoder_finetuned
```