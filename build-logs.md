ubuntu@l40s-180-us-west-or-1:~/qwen3_voice/Qwen3_Voice$ docker-compose up -d
Creating network "voice-gateway" with driver "bridge"
Creating volume "qwen3_voice_omni_model_cache" with local driver
Creating volume "qwen3_voice_tts_model_cache" with local driver
Creating omni_voice_service ... done
Creating tts_read_service   ... done

ERROR: for gateway  Container "0c7e1ed50417" is unhealthy.
ERROR: Encountered errors while bringing up the project.
ubuntu@l40s-180-us-west-or-1:~/qwen3_voice/Qwen3_Voice$ docker ps
CONTAINER ID   IMAGE                                 COMMAND                  CREATED          STATUS                            PORTS                                         NAMES
0c7e1ed50417   qwen3_voice_tts-service               "/opt/nvidia/nvidia_…"   28 seconds ago   Up 9 seconds (health: starting)   8001/tcp                                      tts_read_service
79e7357db0b0   qwen3_voice_omni-service              "python -m vllm.entr…"   28 seconds ago   Restarting (1) 2 seconds ago                                                    omni_voice_service
8dbf15eb35e1   ghcr.io/berriai/litellm:main-latest   "docker/prod_entrypo…"   7 days ago       Up 7 days                         0.0.0.0:4000->4000/tcp, [::]:4000->4000/tcp   aether-gateway
5617c613f933   redis/redis-stack-server:latest       "/entrypoint.sh"         7 days ago       Up 7 days                         0.0.0.0:6390->6379/tcp, [::]:6390->6379/tcp   litellm-redis
11b0aa777c7a   vllm/vllm-openai:latest               "vllm serve --model …"   7 days ago       Up 7 days                         0.0.0.0:8001->8001/tcp, [::]:8001->8001/tcp   qwen3-vl-thinking
ubuntu@l40s-180-us-west-or-1:~/qwen3_voice/Qwen3_Voice$ docker logs -f 0c7e1ed50417

==========
== CUDA ==
==========

CUDA Version 12.1.0

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

*************************
** DEPRECATION NOTICE! **
*************************
THIS IMAGE IS DEPRECATED and is scheduled for DELETION.
    https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/support-policy.md

/usr/local/lib/python3.10/dist-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:
No module named 'vllm._version'
  from vllm.version import __version__ as VLLM_VERSION
INFO 01-31 10:28:40 api_server.py:528] vLLM API server version dev
INFO 01-31 10:28:40 api_server.py:529] args: Namespace(host='0.0.0.0', port=8001, uvicorn_log_level='info', allow_credentials=False, allowed_origins=['*'], allowed_methods=['*'], allowed_headers=['*'], api_key=None, lora_modules=None, prompt_adapters=None, chat_template=None, response_role='assistant', ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_cert_reqs=0, root_path=None, middleware=[], return_tokens_as_token_ids=False, disable_frontend_multiprocessing=False, enable_auto_tool_choice=False, tool_call_parser=None, tool_parser_plugin='', model='Qwen/Qwen3-TTS-12Hz-0.6B-Base', tokenizer=None, skip_tokenizer_init=False, revision=None, code_revision=None, tokenizer_revision=None, tokenizer_mode='auto', trust_remote_code=False, download_dir=None, load_format='auto', config_format='auto', dtype='float16', kv_cache_dtype='auto', quantization_param_path=None, max_model_len=4096, guided_decoding_backend='outlines', distributed_executor_backend=None, worker_use_ray=False, pipeline_parallel_size=1, tensor_parallel_size=1, max_parallel_loading_workers=None, ray_workers_use_nsight=False, block_size=16, enable_prefix_caching=False, disable_sliding_window=False, use_v2_block_manager=True, num_lookahead_slots=0, seed=0, swap_space=4, cpu_offload_gb=0, gpu_memory_utilization=0.1, num_gpu_blocks_override=None, max_num_batched_tokens=None, max_num_seqs=256, max_logprobs=20, disable_log_stats=False, quantization=None, rope_scaling=None, rope_theta=None, enforce_eager=False, max_context_len_to_capture=None, max_seq_len_to_capture=8192, disable_custom_all_reduce=False, tokenizer_pool_size=0, tokenizer_pool_type='ray', tokenizer_pool_extra_config=None, limit_mm_per_prompt=None, mm_processor_kwargs=None, enable_lora=False, max_loras=1, max_lora_rank=16, lora_extra_vocab_size=256, lora_dtype='auto', long_lora_scaling_factors=None, max_cpu_loras=None, fully_sharded_loras=False, enable_prompt_adapter=False, max_prompt_adapters=1, max_prompt_adapter_token=0, device='auto', num_scheduler_steps=1, multi_step_stream_outputs=True, scheduler_delay_factor=0.0, enable_chunked_prefill=None, speculative_model=None, speculative_model_quantization=None, num_speculative_tokens=None, speculative_disable_mqa_scorer=False, speculative_draft_tensor_parallel_size=None, speculative_max_model_len=None, speculative_disable_by_batch_size=None, ngram_prompt_lookup_max=None, ngram_prompt_lookup_min=None, spec_decoding_acceptance_method='rejection_sampler', typical_acceptance_sampler_posterior_threshold=None, typical_acceptance_sampler_posterior_alpha=None, disable_logprobs_during_spec_decoding=None, model_loader_extra_config=None, ignore_patterns=[], preemption_mode=None, served_model_name=None, qlora_adapter_name_or_path=None, otlp_traces_endpoint=None, collect_detailed_traces=None, disable_async_output_proc=False, override_neuron_config=None, scheduling_policy='fcfs', disable_log_requests=False, max_log_len=None, disable_fastapi_docs=False)
INFO 01-31 10:28:40 api_server.py:166] Multiprocessing frontend to use ipc:///tmp/74c9debc-01e3-4977-82cc-2324521a6400 for IPC Path.
INFO 01-31 10:28:40 api_server.py:179] Started engine process with PID 99
Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/auto/configuration_auto.py", line 1384, in from_pretrained
    config_class = CONFIG_MAPPING[config_dict["model_type"]]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/auto/configuration_auto.py", line 1087, in __getitem__
    raise KeyError(key)
KeyError: 'qwen3_tts'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/usr/local/lib/python3.10/dist-packages/vllm/entrypoints/openai/api_server.py", line 585, in <module>
    uvloop.run(run_server(args))
  File "/usr/local/lib/python3.10/dist-packages/uvloop/__init__.py", line 69, in run
    return loop.run_until_complete(wrapper())
  File "uvloop/loop.pyx", line 1518, in uvloop.loop.Loop.run_until_complete
  File "/usr/local/lib/python3.10/dist-packages/uvloop/__init__.py", line 48, in wrapper
    return await main
  File "/usr/local/lib/python3.10/dist-packages/vllm/entrypoints/openai/api_server.py", line 552, in run_server
    async with build_async_engine_client(args) as engine_client:
  File "/usr/lib/python3.10/contextlib.py", line 199, in __aenter__
    return await anext(self.gen)
  File "/usr/local/lib/python3.10/dist-packages/vllm/entrypoints/openai/api_server.py", line 107, in build_async_engine_client
    async with build_async_engine_client_from_engine_args(
  File "/usr/lib/python3.10/contextlib.py", line 199, in __aenter__
    return await anext(self.gen)
  File "/usr/local/lib/python3.10/dist-packages/vllm/entrypoints/openai/api_server.py", line 184, in build_async_engine_client_from_engine_args
    engine_config = engine_args.create_engine_config()
  File "/usr/local/lib/python3.10/dist-packages/vllm/engine/arg_utils.py", line 900, in create_engine_config
    model_config = self.create_model_config()
  File "/usr/local/lib/python3.10/dist-packages/vllm/engine/arg_utils.py", line 837, in create_model_config
    return ModelConfig(
  File "/usr/local/lib/python3.10/dist-packages/vllm/config.py", line 162, in __init__
    self.hf_config = get_config(self.model, trust_remote_code, revision,
  File "/usr/local/lib/python3.10/dist-packages/vllm/transformers_utils/config.py", line 166, in get_config
    raise e
  File "/usr/local/lib/python3.10/dist-packages/vllm/transformers_utils/config.py", line 147, in get_config
    config = AutoConfig.from_pretrained(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/auto/configuration_auto.py", line 1386, in from_pretrained
    raise ValueError(
ValueError: The checkpoint you are trying to load has model type `qwen3_tts` but Transformers does not recognize this architecture. This could be because of an issue with the checkpoint, or because your version of Transformers is out of date.

You can update Transformers with the command `pip install --upgrade transformers`. If this does not work, and the checkpoint is very new, then there may not be a release version that supports this model yet. In this case, you can get the most up-to-date code by installing Transformers from source with the command `pip install git+https://github.com/huggingface/transformers.git`
/usr/local/lib/python3.10/dist-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:
No module named 'vllm._version'
  from vllm.version import __version__ as VLLM_VERSION
Process SpawnProcess-1:
Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/auto/configuration_auto.py", line 1384, in from_pretrained
    config_class = CONFIG_MAPPING[config_dict["model_type"]]
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/auto/configuration_auto.py", line 1087, in __getitem__
    raise KeyError(key)
KeyError: 'qwen3_tts'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/lib/python3.10/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "/usr/lib/python3.10/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/usr/local/lib/python3.10/dist-packages/vllm/engine/multiprocessing/engine.py", line 392, in run_mp_engine
    engine = MQLLMEngine.from_engine_args(engine_args=engine_args,
  File "/usr/local/lib/python3.10/dist-packages/vllm/engine/multiprocessing/engine.py", line 137, in from_engine_args
    engine_config = engine_args.create_engine_config()
  File "/usr/local/lib/python3.10/dist-packages/vllm/engine/arg_utils.py", line 900, in create_engine_config
    model_config = self.create_model_config()
  File "/usr/local/lib/python3.10/dist-packages/vllm/engine/arg_utils.py", line 837, in create_model_config
    return ModelConfig(
  File "/usr/local/lib/python3.10/dist-packages/vllm/config.py", line 162, in __init__
    self.hf_config = get_config(self.model, trust_remote_code, revision,
  File "/usr/local/lib/python3.10/dist-packages/vllm/transformers_utils/config.py", line 166, in get_config
    raise e
  File "/usr/local/lib/python3.10/dist-packages/vllm/transformers_utils/config.py", line 147, in get_config
    config = AutoConfig.from_pretrained(
  File "/usr/local/lib/python3.10/dist-packages/transformers/models/auto/configuration_auto.py", line 1386, in from_pretrained
    raise ValueError(
ValueError: The checkpoint you are trying to load has model type `qwen3_tts` but Transformers does not recognize this architecture. This could be because of an issue with the checkpoint, or because your version of Transformers is out of date.

You can update Transformers with the command `pip install --upgrade transformers`. If this does not work, and the checkpoint is very new, then there may not be a release version that supports this model yet. In this case, you can get the most up-to-date code by installing Transformers from source with the command `pip install git+https://github.com/huggingface/transformers.git`

==========
== CUDA ==
==========

CUDA Version 12.1.0

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

*************************
** DEPRECATION NOTICE! **
*************************
THIS IMAGE IS DEPRECATED and is scheduled for DELETION.
    https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/support-policy.md

/usr/local/lib/python3.10/dist-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:
No module named 'vllm._version'
