This repository uses `uv` to run. Instructions to install [here](https://docs.astral.sh/uv/getting-started/installation/).

## Preparing the datasets

### Image-Caption extraction from PDFs
**RUN TIME:** ~3min

Extract images and captions from both PDFs with:
```bash
uv run extract_image_caption.py source_books/textbook_of_pathology.pdf --pages 1000 --output-dir test_output
uv run extract_image_caption.py source_books/basic_pathology.pdf --pages 1000 --output-dir test_output
```

The resulting folders will have the following structure:
```bash
$ tree
.
├── basic_pathology
│   └── basic_pathology
│       ├── images.csv
│       ├── img_001.jpg
│       └── ...
└── textbook_of_pathology
    └── textbook_of_pathology
        ├── images.csv
        ├── img_001.jpg
        └── ...

4 directories, 2933 files
```
More files will be added later in the pipeline, so the count should be a few dozens smaller than 2933.

The results from the original run were saved under `output`.
Finally, we filter out images whose code could not find the caption, samples with non unique captions (meaning that the code could not split the text for subfigures), images with wrong sizing, and other filters. Execute
```bash
uv run filter_image_caption.py --dataset-dir test_output/textbook_of_pathology/ --remove-dir remove_images
uv run filter_image_caption.py --dataset-dir test_output/basic_pathology/ --remove-dir remove_images
```

### EntiGraph
**RUN TIME (COMPLETE DATASET):** ~12h using 20 machines with 20GB of VRAM each

**RUN TIME (SINGLE DOCUMENT):** ~1h (but you can interrupt the code after ~5min for partial results)

> [!NOTE]
> The raw data generated from this step can be found on the `output` folder of this repository. The corresponding dataset can be downloaded from [huggingface](https://huggingface.co/datasets/joao-donasolo/entigraph-pathvqa).

Now, we run EntiGraph to generate the synthetic dataset. For each image, the code will 1) Extract relevant entities from the image/caption pair, and 2) reason about their relationship in groups of two. Each reasoning produces a new synthetic document. Therefore, an image with 5 extracted entities will generate C(5,2) = 10 new documents. On the original paper, the authors sample k relationships, but here we did not need to do so, and used the full dataset for training.
The code uses Qwen3.5-9B and about 20GB of VRAM. You can use the `--start` and `--end` arguments to make the run parallel across multiple machines. To execute the pipeline for a single document, run
```bash
uv run src/extract_entities.py --images-dir test_output/textbook_of_pathology/ --cache-dir ./huggingface --start 0 --end 1
```

Entities will be saved under `test_output/textbook_of_pathology/entities.jsonl`, and relationships under `test_output/textbook_of_pathology/relations.jsonl`. After 5 minutes, you should already be able to see a couple of examples.
The results from the original run were saved under `output`.

## Training the model
**RUN TIME:** ~44h

> [!NOTE]
> The final model can be downloaded from [huggingface](https://huggingface.co/joao-donasolo/entigraph-pathvqa).

Now, we proceed to the training of the model. The best configuration found is saved at `configs/instruct/instruct_withvqa_lora32_lr1.0e-5_bs128_full.yaml`, and uses the full synthetic dataset (both source textbooks) and PathVQA training samples. To train the model, execute the command below. It will download the base Qwen model and the PathVQA dataset, generate the EntiGraph dataset from the output folder, and run the training with the specified configurations.
```bash
uv run src/train_instruct_with_augment  ation.py -c configs/instruct/instruct_withvqa_lora32_lr1.0e-5_bs128_full.yaml
```

The code uses weights and biases to report the metrics in a friendly interface. To disable it, add `report_to: none` in the yaml file, under the training section.

## Evaluation
**RUN TIME:** between 15min and 20min per run

To evaluate the pretrained instruct model, run
```bash
uv run src/evaluate.py --model "Qwen/Qwen3.5-0.8B" --batch-size 64 --output eval_results.json
```

To evaluate the our final model, run
```bash
uv run src/evaluate.py --model "Qwen/Qwen3.5-0.8B" --checkpoint joao-donasolo/entigraph-pathvqa --batch-size 64 --output eval_results.json
```