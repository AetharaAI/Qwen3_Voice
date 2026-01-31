 Created wheel for webrtcvad: filename=webrtcvad-2.0.10-cp311-cp311-linux_x86_64.whl size=73587 sha256=5d74055fbe3fc6a36f1d8ad4c9142bc99da5e9fc6d3c641ef88c7ae84cd4d4bd
  Stored in directory: /tmp/pip-ephem-wheel-cache-9vltcme_/wheels/94/65/3f/292d0b656be33d1c801831201c74b5f68f41a2ae465ff2ee2f
Successfully built webrtcvad
Installing collected packages: webrtcvad, nvidia-ml-py, websockets, uvloop, urllib3, typing-extensions, threadpoolctl, tenacity, structlog, sniffio, redis, pyyaml, python-multipart, python-json-logger, python-dotenv, pynvml, pycparser, propcache, prometheus-client, platformdirs, orjson, numpy, multidict, msgpack, llvmlite, lazy-loader, joblib, idna, hyperframe, httptools, hpack, h11, frozenlist, decorator, click, charset_normalizer, certifi, audioread, attrs, annotated-types, aiohappyeyeballs, yarl, uvicorn, soxr, scipy, requests, pydantic-core, numba, httpcore, h2, cffi, anyio, aiosignal, watchfiles, starlette, soundfile, scikit-learn, pydantic, pooch, httpx, aiohttp, starlette-prometheus, pydantic-settings, librosa, fastapi

Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.11.0 aiosignal-1.4.0 annotated-types-0.7.0 anyio-4.12.1 attrs-25.4.0 audioread-3.1.0 certifi-2026.1.4 cffi-2.0.0 charset_normalizer-3.4.4 click-8.3.1 decorator-5.2.1 fastapi-0.115.0 frozenlist-1.8.0 h11-0.16.0 h2-4.3.0 hpack-4.1.0 httpcore-1.0.9 httptools-0.7.1 httpx-0.27.2 hyperframe-6.1.0 idna-3.11 joblib-1.5.3 lazy-loader-0.4 librosa-0.10.2.post1 llvmlite-0.46.0 msgpack-1.1.2 multidict-6.7.1 numba-0.63.1 numpy-1.26.4 nvidia-ml-py-12.560.30 orjson-3.10.11 platformdirs-4.5.1 pooch-1.9.0 prometheus-client-0.21.0 propcache-0.4.1 pycparser-3.0 pydantic-2.9.2 pydantic-core-2.23.4 pydantic-settings-2.6.1 pynvml-11.5.0 python-dotenv-1.2.1 python-json-logger-2.0.7 python-multipart-0.0.12 pyyaml-6.0.3 redis-5.2.0 requests-2.32.5 scikit-learn-1.8.0 scipy-1.14.1 sniffio-1.3.1 soundfile-0.12.1 soxr-1.0.0 starlette-0.38.6 starlette-prometheus-0.10.0 structlog-24.4.0 tenacity-9.0.0 threadpoolctl-3.6.0 typing-extensions-4.15.0 urllib3-2.6.3 uvicorn-0.32.0 uvloop-0.22.1 watchfiles-1.1.1 webrtcvad-2.0.10 websockets-13.1 yarl-1.22.0
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
 ---> Removed intermediate container 52305bdde1cf
 ---> b3540a499002
Step 10/18 : COPY . .
 ---> a6a61f4934d4
Step 11/18 : RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
 ---> Running in 20ca7704d750
 ---> Removed intermediate container 20ca7704d750
 ---> 343c25542621
Step 12/18 : USER appuser
 ---> Running in 8d0e5de016f6
 ---> Removed intermediate container 8d0e5de016f6
 ---> 0af57f3149df
Step 13/18 : ENV PYTHONDONTWRITEBYTECODE=1
 ---> Running in db143e2de1ca
 ---> Removed intermediate container db143e2de1ca
 ---> fa6ce94564a0
Step 14/18 : ENV PYTHONUNBUFFERED=1
 ---> Running in be2049f808db
 ---> Removed intermediate container be2049f808db
 ---> 5e09fd23d1b8
Step 15/18 : ENV PYTHONPATH=/app
 ---> Running in b61f07fc2779
 ---> Removed intermediate container b61f07fc2779
 ---> 4b97e53a49fd
Step 16/18 : EXPOSE 8000
 ---> Running in f599b64b83f9
 ---> Removed intermediate container f599b64b83f9
 ---> 49d004f266c4
Step 17/18 : HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3     CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
 ---> Running in b6038b164df6
 ---> Removed intermediate container b6038b164df6
 ---> e68b399a8633
Step 18/18 : CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--loop", "uvloop"]
 ---> Running in b965ddf88c91
 ---> Removed intermediate container b965ddf88c91
 ---> 62a967820ffa
Successfully built 62a967820ffa
Successfully tagged qwen3_voice_gateway:latest
ubuntu@l40s-180-us-west-or-1:~/qwen3_voice/Qwen3_Voice$ docker-compose up -d
Creating network "voice-gateway" with driver "bridge"
Creating tts_read_service   ... done
Creating omni_voice_service ... done

ERROR: for gateway  Container "f7db8971426a" is unhealthy.
ERROR: Encountered errors while bringing up the project.
ubuntu@l40s-180-us-west-or-1:~/qwen3_voice/Qwen3_Voice$ docker logs -f f7db8971426a

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
INFO 01-31 12:41:34 api_server.py:528] vLLM API server version dev
INFO 01-31 12:41:34 api_server.py:529] args: Namespace(host='0.0.0.0', port=8001, uvicorn_log_level='info', allow_credentials=False, allowed_origins=['*'], allowed_methods=['*'], allowed_headers=['*'], api_key=None, lora_modules=None, prompt_adapters=None, chat_template=None, response_role='assistant', ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_cert_reqs=0, root_path=None, middleware=[], return_tokens_as_token_ids=False, disable_frontend_multiprocessing=False, enable_auto_tool_choice=False, tool_call_parser=None, tool_parser_plugin='', model='Qwen/Qwen3-TTS-12Hz-0.6B-Base', tokenizer=None, skip_tokenizer_init=False, revision=None, code_revision=None, tokenizer_revision=None, tokenizer_mode='auto', trust_remote_code=False, download_dir=None, load_format='auto', config_format='auto', dtype='bfloat16', kv_cache_dtype='auto', quantization_param_path=None, max_model_len=4096, guided_decoding_backend='outlines', distributed_executor_backend=None, worker_use_ray=False, pipeline_parallel_size=1, tensor_parallel_size=1, max_parallel_loading_workers=None, ray_workers_use_nsight=False, block_size=16, enable_prefix_caching=False, disable_sliding_window=False, use_v2_block_manager=True, num_lookahead_slots=0, seed=0, swap_space=4, cpu_offload_gb=0, gpu_memory_utilization=0.1, num_gpu_blocks_override=None, max_num_batched_tokens=None, max_num_seqs=256, max_logprobs=20, disable_log_stats=False, quantization=None, rope_scaling=None, rope_theta=None, enforce_eager=False, max_context_len_to_capture=None, max_seq_len_to_capture=8192, disable_custom_all_reduce=False, tokenizer_pool_size=0, tokenizer_pool_type='ray', tokenizer_pool_extra_config=None, limit_mm_per_prompt=None, mm_processor_kwargs=None, enable_lora=False, max_loras=1, max_lora_rank=16, lora_extra_vocab_size=256, lora_dtype='auto', long_lora_scaling_factors=None, max_cpu_loras=None, fully_sharded_loras=False, enable_prompt_adapter=False, max_prompt_adapters=1, max_prompt_adapter_token=0, device='auto', num_scheduler_steps=1, multi_step_stream_outputs=True, scheduler_delay_factor=0.0, enable_chunked_prefill=None, speculative_model=None, speculative_model_quantization=None, num_speculative_tokens=None, speculative_disable_mqa_scorer=False, speculative_draft_tensor_parallel_size=None, speculative_max_model_len=None, speculative_disable_by_batch_size=None, ngram_prompt_lookup_max=None, ngram_prompt_lookup_min=None, spec_decoding_acceptance_method='rejection_sampler', typical_acceptance_sampler_posterior_threshold=None, typical_acceptance_sampler_posterior_alpha=None, disable_logprobs_during_spec_decoding=None, model_loader_extra_config=None, ignore_patterns=[], preemption_mode=None, served_model_name=None, qlora_adapter_name_or_path=None, otlp_traces_endpoint=None, collect_detailed_traces=None, disable_async_output_proc=False, override_neuron_config=None, scheduling_policy='fcfs', disable_log_requests=False, max_log_len=None, disable_fastapi_docs=False)
INFO 01-31 12:41:34 api_server.py:166] Multiprocessing frontend to use ipc:///tmp/7a8a96b7-d0af-4f54-a3a3-e4c248ddd459 for IPC Path.
INFO 01-31 12:41:34 api_server.py:179] Started engine process with PID 93
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
  from vllm.version import __version__ as VLLM_VERSION
INFO 01-31 12:41:51 api_server.py:528] vLLM API server version dev
INFO 01-31 12:41:51 api_server.py:529] args: Namespace(host='0.0.0.0', port=8001, uvicorn_log_level='info', allow_credentials=False, allowed_origins=['*'], allowed_methods=['*'], allowed_headers=['*'], api_key=None, lora_modules=None, prompt_adapters=None, chat_template=None, response_role='assistant', ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_cert_reqs=0, root_path=None, middleware=[], return_tokens_as_token_ids=False, disable_frontend_multiprocessing=False, enable_auto_tool_choice=False, tool_call_parser=None, tool_parser_plugin='', model='Qwen/Qwen3-TTS-12Hz-0.6B-Base', tokenizer=None, skip_tokenizer_init=False, revision=None, code_revision=None, tokenizer_revision=None, tokenizer_mode='auto', trust_remote_code=False, download_dir=None, load_format='auto', config_format='auto', dtype='bfloat16', kv_cache_dtype='auto', quantization_param_path=None, max_model_len=4096, guided_decoding_backend='outlines', distributed_executor_backend=None, worker_use_ray=False, pipeline_parallel_size=1, tensor_parallel_size=1, max_parallel_loading_workers=None, ray_workers_use_nsight=False, block_size=16, enable_prefix_caching=False, disable_sliding_window=False, use_v2_block_manager=True, num_lookahead_slots=0, seed=0, swap_space=4, cpu_offload_gb=0, gpu_memory_utilization=0.1, num_gpu_blocks_override=None, max_num_batched_tokens=None, max_num_seqs=256, max_logprobs=20, disable_log_stats=False, quantization=None, rope_scaling=None, rope_theta=None, enforce_eager=False, max_context_len_to_capture=None, max_seq_len_to_capture=8192, disable_custom_all_reduce=False, tokenizer_pool_size=0, tokenizer_pool_type='ray', tokenizer_pool_extra_config=None, limit_mm_per_prompt=None, mm_processor_kwargs=None, enable_lora=False, max_loras=1, max_lora_rank=16, lora_extra_vocab_size=256, lora_dtype='auto', long_lora_scaling_factors=None, max_cpu_loras=None, fully_sharded_loras=False, enable_prompt_adapter=False, max_prompt_adapters=1, max_prompt_adapter_token=0, device='auto', num_scheduler_steps=1, multi_step_stream_outputs=True, scheduler_delay_factor=0.0, enable_chunked_prefill=None, speculative_model=None, speculative_model_quantization=None, num_speculative_tokens=None, speculative_disable_mqa_scorer=False, speculative_draft_tensor_parallel_size=None, speculative_max_model_len=None, speculative_disable_by_batch_size=None, ngram_prompt_lookup_max=None, ngram_prompt_lookup_min=None, spec_decoding_acceptance_method='rejection_sampler', typical_acceptance_sampler_posterior_threshold=None, typical_acceptance_sampler_posterior_alpha=None, disable_logprobs_during_spec_decoding=None, model_loader_extra_config=None, ignore_patterns=[], preemption_mode=None, served_model_name=None, qlora_adapter_name_or_path=None, otlp_traces_endpoint=None, collect_detailed_traces=None, disable_async_output_proc=False, override_neuron_config=None, scheduling_policy='fcfs', disable_log_requests=False, max_log_len=None, disable_fastapi_docs=False)
INFO 01-31 12:41:51 api_server.py:166] Multiprocessing frontend to use ipc:///tmp/8c0dccce-cbb9-471e-95c0-e9cb6662cecf for IPC Path.
INFO 01-31 12:41:51 api_server.py:179] Started engine process with PID 99
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
  from vllm.version import __version__ as VLLM_VERSION
INFO 01-31 12:42:12 api_server.py:528] vLLM API server version dev
INFO 01-31 12:42:12 api_server.py:529] args: Namespace(host='0.0.0.0', port=8001, uvicorn_log_level='info', allow_credentials=False, allowed_origins=['*'], allowed_methods=['*'], allowed_headers=['*'], api_key=None, lora_modules=None, prompt_adapters=None, chat_template=None, response_role='assistant', ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_cert_reqs=0, root_path=None, middleware=[], return_tokens_as_token_ids=False, disable_frontend_multiprocessing=False, enable_auto_tool_choice=False, tool_call_parser=None, tool_parser_plugin='', model='Qwen/Qwen3-TTS-12Hz-0.6B-Base', tokenizer=None, skip_tokenizer_init=False, revision=None, code_revision=None, tokenizer_revision=None, tokenizer_mode='auto', trust_remote_code=False, download_dir=None, load_format='auto', config_format='auto', dtype='bfloat16', kv_cache_dtype='auto', quantization_param_path=None, max_model_len=4096, guided_decoding_backend='outlines', distributed_executor_backend=None, worker_use_ray=False, pipeline_parallel_size=1, tensor_parallel_size=1, max_parallel_loading_workers=None, ray_workers_use_nsight=False, block_size=16, enable_prefix_caching=False, disable_sliding_window=False, use_v2_block_manager=True, num_lookahead_slots=0, seed=0, swap_space=4, cpu_offload_gb=0, gpu_memory_utilization=0.1, num_gpu_blocks_override=None, max_num_batched_tokens=None, max_num_seqs=256, max_logprobs=20, disable_log_stats=False, quantization=None, rope_scaling=None, rope_theta=None, enforce_eager=False, max_context_len_to_capture=None, max_seq_len_to_capture=8192, disable_custom_all_reduce=False, tokenizer_pool_size=0, tokenizer_pool_type='ray', tokenizer_pool_extra_config=None, limit_mm_per_prompt=None, mm_processor_kwargs=None, enable_lora=False, max_loras=1, max_lora_rank=16, lora_extra_vocab_size=256, lora_dtype='auto', long_lora_scaling_factors=None, max_cpu_loras=None, fully_sharded_loras=False, enable_prompt_adapter=False, max_prompt_adapters=1, max_prompt_adapter_token=0, device='auto', num_scheduler_steps=1, multi_step_stream_outputs=True, scheduler_delay_factor=0.0, enable_chunked_prefill=None, speculative_model=None, speculative_model_quantization=None, num_speculative_tokens=None, speculative_disable_mqa_scorer=False, speculative_draft_tensor_parallel_size=None, speculative_max_model_len=None, speculative_disable_by_batch_size=None, ngram_prompt_lookup_max=None, ngram_prompt_lookup_min=None, spec_decoding_acceptance_method='rejection_sampler', typical_acceptance_sampler_posterior_threshold=None, typical_acceptance_sampler_posterior_alpha=None, disable_logprobs_during_spec_decoding=None, model_loader_extra_config=None, ignore_patterns=[], preemption_mode=None, served_model_name=None, qlora_adapter_name_or_path=None, otlp_traces_endpoint=None, collect_detailed_traces=None, disable_async_output_proc=False, override_neuron_config=None, scheduling_policy='fcfs', disable_log_requests=False, max_log_len=None, disable_fastapi_docs=False)
INFO 01-31 12:42:12 api_server.py:166] Multiprocessing frontend to use ipc:///tmp/11c8da07-40e6-4446-b1a2-3407139bf14b for IPC Path.
INFO 01-31 12:42:12 api_server.py:179] Started engine process with PID 99
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
  File "/usr/local/lib/python3.10/dist-pac