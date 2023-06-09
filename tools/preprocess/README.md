# Pre-Process


## Generate FH masks for COCO Datasets

As shown in the repository, the datasets are assumed to exist in a directory specified by the environment variable $E2E_DATASETS.
In order to make it consistent, we want to generate FH mask proposals and save them to ```fh_train2017``` and ```fh_unlabeled2017``` folders under $E2E_DATASETS.

```
$E2E_DATASETS/
└── coco/
	├── annotations/
		├── instances_train2017.json
		├── image_info_unlabeled2017.json
		├── instances_val2017.json
		└── image_info_test-dev2017.json
	├── image/
		├── train2017/
		├── fh_train2017/
		├── unlabeled2017/
		├── fh_unlabeled2017/
		├── val2017/
		└── test2017/
	└── vocabs/
		└── coco_categories.txt - the mapping from coco categories to indices.
├── imagenet/
	├── fh_train/
	├── fh_val/
```

The command for generating ```fh_train2017``` is as following:
```bash
python create_fh_mask_for_coco.py --root_path $E2E_DATASETS/coco/image --image_folder train2017 --output_folder fh_train2017 --fh_scales '500,1000,1500' --fh_min_sizes '500,1000,1500'
```

The command for generating ```fh_unlabeled2017``` is as following:
```bash
python create_fh_mask_for_coco.py --root_path $E2E_DATASETS/coco/image --image_folder unlabeled2017 --output_folder fh_unlabeled2017 --fh_scales '500,1000,1500' --fh_min_sizes '500,1000,1500'
```


## Generate FH masks for ImageNet Datasets

As shown in the repository, the datasets are assumed to exist in a directory specified by the environment variable $E2E_DATASETS.
In order to make it consistent, we want to generate FH mask proposals and save them to ```fh_train``` and ```fh_val``` folders under $E2E_DATASETS.

```
$E2E_DATASETS/
└── coco/
	├── annotations/
		├── instances_train2017.json
		├── image_info_unlabeled2017.json
		├── instances_val2017.json
		├── sam_instances_train2017.json
		├── sam_image_info_unlabeled2017.json
		├── sam_instances_val2017.json
		└── image_info_test-dev2017.json
	├── image/
		├── train2017/
		├── unlabeled2017/
		├── val2017/
		└── test2017/
	└── vocabs/
		└── coco_categories.txt - the mapping from coco categories to indices.
├── imagenet/
	├── sam_train/
	├── sam_val/
```

The command for generating ```fh_train``` is as following:
```bash
python create_fh_mask_for_imnet.py --root_path /datasets01/imagenet_full_size/061417 --image_folder train --output_path $HOME/proposal/data/imnet --output_folder fh_train --fh_scales '1000' --fh_min_sizes '1000'
```
