---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:19872
- loss:CombinedLoss
base_model: kuelumbus/polyBERT
widget:
- source_sentence: '*C1(C(=O)C(CCC1)C*)CO'
  sentences:
  - '*Oc1ccc(cc1)OC(=O)c1ccc(cc1)NC(=O)c1ccc(cc1)C(=O)*'
  - '[*]Oc1cc(CCCCCCCCCCCC)c([*])cc1Nc1ccc(CCCCCCCCCCCC)cc1C'
  - '*OC(=O)C1C(=O)CC(C(=O)C1)C(=O)OCCCCCC*'
- source_sentence: '[*]Nc1cc(-c2ccc(C#N)cc2)cc(-c2ccc(-c3cccc(-c4ccn([*])n4)c3)cc2)c1'
  sentences:
  - '*SCSCCCC*'
  - '[*]OCOc1cc(C)ccc1-c1ccc(C)cc1OCc1cc(C)ccc1O[*]'
  - '[*]Oc1cccc(C(=O)OCCNc2ccc(S(=O)(=O)c3ccc([*])cc3)cc2)c1'
- source_sentence: '*C(C(CC*)(F)F)(Cl)F'
  sentences:
  - '[*]Oc1cc([*])ccc1Nc1cc(-c2ccc(N=Nc3ccc(C#N)cc3)cc2)ccc1CCCCCCCCC'
  - '[*]C#CC#CCOc1ccc(NC(=O)OC[*])cc1C'
  - '*Oc1ccc(cc1)N=Nc1ccc(cc1)*'
- source_sentence: '*C(=C(*)C)[Si](CC)(C)C'
  sentences:
  - '[*]CCCCCCCCCOc1ccc(Cc2ccc(C[*])cc2)o1'
  - '[*]n1c2ccccc2c2cc(-c3ccc4c(c3)c3ccccc3n4-c3cc(-c4ccc(C)c(OCC[N+]([*])(C)C)c4)ccc3C)ccc21'
  - '[*]Nc1cc(C(=O)Oc2ccc(O[*])nc2)cc(-c2ccccn2)c1'
- source_sentence: '*NC(C(=O)*)CCC(=O)OCCCCCCCCCCCC'
  sentences:
  - '*O[Si](*)(CCC(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)C'
  - '[*]NC(=O)CCCCCCCCCCCCCCC(=O)OC1C(=O)N(c2ccc3nc(CCCCCC)c(-c4ccc([*])cc4)nc3c2)C(=O)C1C'
  - '[*]c1ccc(C2(NC3CCC(C)(Oc4cc([*])[nH]n4)CC3)CC3CC2C2CCCC32)cc1'
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
    '*NC(C(=O)*)CCC(=O)OCCCCCCCCCCCC',
    '*O[Si](*)(CCC(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)C',
    '[*]NC(=O)CCCCCCCCCCCCCCC(=O)OC1C(=O)N(c2ccc3nc(CCCCCC)c(-c4ccc([*])cc4)nc3c2)C(=O)C1C',
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

* Size: 19,872 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                         | label                              |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------|
  | type    | string                                                                             | string                                                                             | list                               |
  | details | <ul><li>min: 9 tokens</li><li>mean: 56.12 tokens</li><li>max: 212 tokens</li></ul> | <ul><li>min: 12 tokens</li><li>mean: 64.5 tokens</li><li>max: 212 tokens</li></ul> | <ul><li>size: 2 elements</li></ul> |
* Samples:
  | sentence_0                                             | sentence_1                                                                                         | label                                                 |
  |:-------------------------------------------------------|:---------------------------------------------------------------------------------------------------|:------------------------------------------------------|
  | <code>*c1c2c(nccc2)c(cc1)OCCOc1c2ncccc2c(cc1)C*</code> | <code>[*]c1ncc(-c2c(Oc3ccc4ccccc4c3-c3ccc4ccccc4c3-c3cnc([*])c(C)c3)ccc3ccccc23)cc1C</code>        | <code>[0.8013490436761502, 0.396903507376926]</code>  |
  | <code>*NNC(=O)CCC(=O)NNC(=O)CCCCCCCCC(=O)*</code>      | <code>[*]NC(=O)CCCCCCCCCCCCCCC(=O)OC1C(=O)N(c2ccc3nc(CCCCCC)c(-c4ccc([*])cc4)nc3c2)C(=O)C1C</code> | <code>[0.9219285075777384, 0.6954722618099292]</code> |
  | <code>*c1[nH]c(cc1CC(=O)OCCCCCCCC)*</code>             | <code>[*]OC1(S[*])CCCCCCCCCCC1</code>                                                              | <code>[0.8822615410705786, 0.5782884620801875]</code> |
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
| 0.4026 | 500  | 0.0291        |
| 0.8052 | 1000 | 0.007         |
| 1.2077 | 1500 | 0.0034        |
| 1.6103 | 2000 | 0.0024        |
| 2.0129 | 2500 | 0.0019        |
| 2.4155 | 3000 | 0.0015        |
| 2.8180 | 3500 | 0.0013        |
| 3.2206 | 4000 | 0.0011        |
| 3.6232 | 4500 | 0.001         |
| 4.0258 | 5000 | 0.0009        |
| 4.4283 | 5500 | 0.0008        |
| 4.8309 | 6000 | 0.0008        |


### Framework Versions
- Python: 3.11.9
- Sentence Transformers: 3.4.1
- Transformers: 4.49.0
- PyTorch: 2.6.0+cu124
- Accelerate: 1.4.0
- Datasets: 3.3.2
- Tokenizers: 0.21.0
