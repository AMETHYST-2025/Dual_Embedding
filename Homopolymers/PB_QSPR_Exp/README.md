---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:19840
- loss:CombinedLoss
base_model: kuelumbus/polyBERT
widget:
- source_sentence: CC(COC(=O)C(=C)C)O
  sentences:
  - CC(=C)C(=O)OCCO
  - Oc1ccc(c2ccccc12)S(=O)(=O)c1ccccc1
  - C(=O)c1ccc(cc1)C(=O)Oc1ccc(cc1)C(C)(C)c1 ccc(cc1)O
- source_sentence: O(c1ccc(cc1)c1cc(nc2ccccc12)c1ccc(cc1)c1ccc cc1)c1ccc(cc1)c1ccnc2ccccc12
  sentences:
  - Oc1ccc(cc1)C1(CCCCC1)c1ccc(Oc2ccc(cc2)S (=O)(=O)c2ccccc2)cc1
  - 'Cc1cc(ccc1O)C(C)(C)c1cc(C)c(Oc2ccc(cc2)S(

    =O)(=O)c2ccccc2)cc1'
  - O[SiH](C)c1ccccc1
- source_sentence: OC(=O)C1=C(C=C)C=CC=C1
  sentences:
  - C=CC1=C(COC(=O)C2=CC=CC=C2)C=CC= C1
  - CCC=C(C)C([O-])=O
  - CC1(C2CCC1(C(C2)OC(=O)C=C)C)C
- source_sentence: O=C1OCCOC(=O)c2cc3c(cc2)cc1cc3
  sentences:
  - C(Cl)(C(=O)OCCC)C
  - CC(C)CCOC(=O)C=C
  - C=CC1=CC=CC=C1
- source_sentence: C1C(O1)CCl
  sentences:
  - 'COC1=CC=C(C=C1)C(=O)C1=CC=C(C=C)C

    =C1'
  - CCCCC(CC)COC(=O)C=C
  - CCCCC(CC)CCC(CC(C)C)OC(=O)C=C
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on kuelumbus/polyBERT

This is a [sentence-transformers] model finetuned from [kuelumbus/polyBERT]. It maps sentences & paragraphs to a 600-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [kuelumbus/polyBERT] <!-- at revision deaa98fb65a7bdfb537457d42f43bd468963f695 -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 600 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation]
- **Repository:** [Sentence Transformers on GitHub]
- **Hugging Face:** [Sentence Transformers on Hugging Face]

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: DebertaV2Model 
  (1): Pooling({'word_embedding_dimension': 600, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'C1C(O1)CCl',
    'CCCCC(CC)CCC(CC(C)C)OC(=O)C=C',
    'COC1=CC=C(C=C1)C(=O)C1=CC=C(C=C)C\n=C1',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 600]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 19,840 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                         | label                              |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------|
  | type    | string                                                                             | string                                                                             | list                               |
  | details | <ul><li>min: 6 tokens</li><li>mean: 37.72 tokens</li><li>max: 124 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 39.15 tokens</li><li>max: 205 tokens</li></ul> | <ul><li>size: 2 elements</li></ul> |
* Samples:
  | sentence_0                                                                               | sentence_1                                | label                                                 |
  |:-----------------------------------------------------------------------------------------|:------------------------------------------|:------------------------------------------------------|
  | <code>Oc1ccc(cc1)c1ccc(Oc2cc3c(cc2)C(=O)N(C3= O)c2cccc(c2)N2C(=O)c3ccccc3C2=O)cc1</code> | <code>CC(COC(=O)C(=C)C)O</code>           | <code>[0.6606212072820745, 0.2062463527923418]</code> |
  | <code>CC(C)CC(C)(C)CCOC(=O)C(C)=C</code>                                                 | <code>CCCCC(CC)COC(=O)C=C</code>          | <code>[0.8712847968720312, 0.5495020115948286]</code> |
  | <code>CC(=O)C1=CC=C(C=C1)C=C</code>                                                      | <code>C(=O)CCCCC(=O)NCc1cc(ccc1)CN</code> | <code>[0.9484806762930592, 0.7869018352034158]</code> |
* Loss: <code>__main__.CombinedLoss</code>

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 5
- `batch_sampler`: no_duplicates
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 5
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: no_duplicates
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.4032 | 500  | 0.0304        |
| 0.8065 | 1000 | 0.0077        |
| 1.2097 | 1500 | 0.0041        |
| 1.6129 | 2000 | 0.0032        |
| 2.0161 | 2500 | 0.0028        |
| 2.4194 | 3000 | 0.0022        |
| 2.8226 | 3500 | 0.0022        |
| 3.2258 | 4000 | 0.0018        |
| 3.6290 | 4500 | 0.0017        |
| 4.0323 | 5000 | 0.0016        |
| 4.4355 | 5500 | 0.0015        |
| 4.8387 | 6000 | 0.0014        |


### Framework Versions
- Python: 3.11.9
- Sentence Transformers: 3.4.1
- Transformers: 4.49.0
- PyTorch: 2.6.0+cu124
- Accelerate: 1.4.0
- Datasets: 3.3.2
- Tokenizers: 0.21.0
