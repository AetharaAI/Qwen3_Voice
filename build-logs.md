
Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.11.0 aiosignal-1.4.0 annotated-types-0.7.0 anyio-4.12.1 attrs-25.4.0 audioread-3.1.0 certifi-2026.1.4 cffi-2.0.0 charset_normalizer-3.4.4 click-8.3.1 decorator-5.2.1 fastapi-0.115.0 frozenlist-1.8.0 h11-0.16.0 h2-4.3.0 hpack-4.1.0 httpcore-1.0.9 httptools-0.7.1 httpx-0.27.2 hyperframe-6.1.0 idna-3.11 joblib-1.5.3 lazy-loader-0.4 librosa-0.10.2.post1 llvmlite-0.46.0 msgpack-1.1.2 multidict-6.7.1 numba-0.63.1 numpy-1.26.4 nvidia-ml-py-12.560.30 orjson-3.10.11 platformdirs-4.5.1 pooch-1.9.0 prometheus-client-0.21.0 propcache-0.4.1 pycparser-3.0 pydantic-2.9.2 pydantic-core-2.23.4 pydantic-settings-2.6.1 pynvml-11.5.0 python-dotenv-1.2.1 python-json-logger-2.0.7 python-multipart-0.0.12 pyyaml-6.0.3 redis-5.2.0 requests-2.32.5 scikit-learn-1.8.0 scipy-1.14.1 sniffio-1.3.1 soundfile-0.12.1 soxr-1.0.0 starlette-0.38.6 starlette-prometheus-0.10.0 structlog-24.4.0 tenacity-9.0.0 threadpoolctl-3.6.0 typing-extensions-4.15.0 urllib3-2.6.3 uvicorn-0.32.0 uvloop-0.22.1 watchfiles-1.1.1 webrtcvad-2.0.10 websockets-13.1 yarl-1.22.0
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
 ---> Removed intermediate container cad9bf346154
 ---> 862b43dddaaf
Step 10/18 : COPY . .
 ---> c3a8cbc982c4
Step 11/18 : RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
 ---> Running in d9297484dc92
 ---> Removed intermediate container d9297484dc92
 ---> 0b03447289fd
Step 12/18 : USER appuser
 ---> Running in 5ee690bfd66c
 ---> Removed intermediate container 5ee690bfd66c
 ---> 73914cf974e3
Step 13/18 : ENV PYTHONDONTWRITEBYTECODE=1
 ---> Running in c86f5e3d24b6
 ---> Removed intermediate container c86f5e3d24b6
 ---> e563d0d56151
Step 14/18 : ENV PYTHONUNBUFFERED=1
 ---> Running in 02061c3c8b28
 ---> Removed intermediate container 02061c3c8b28
 ---> f62a5228383c
Step 15/18 : ENV PYTHONPATH=/app
 ---> Running in f03061e3145c
 ---> Removed intermediate container f03061e3145c
 ---> ace766028788
Step 16/18 : EXPOSE 8000
 ---> Running in cba117a557be
 ---> Removed intermediate container cba117a557be
 ---> ddac65a397a6
Step 17/18 : HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3     CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
 ---> Running in e4769f28ecb9
 ---> Removed intermediate container e4769f28ecb9
 ---> 8efd0c8b5de1
Step 18/18 : CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--loop", "uvloop"]
 ---> Running in 007f18a97c39
 ---> Removed intermediate container 007f18a97c39
 ---> 8023e56fe9a0
Successfully built 8023e56fe9a0
Successfully tagged qwen3_voice_gateway:latest
ubuntu@l40s-180-us-west-or-1:~/qwen3_voice/Qwen3_Voice$ docker-compose up -d
Creating network "voice-gateway" with driver "bridge"
Creating tts_read_service   ... done
Creating omni_voice_service ... done

ERROR: for gateway  Container "10134a8b674d" is unhealthy.
ERROR: Encountered errors while bringing up the project.
ubuntu@l40s-180-us-west-or-1:~/qwen3_voice/Qwen3_Voice$ docker logs -f voice-gateway
Error response from daemon: No such container: voice-gateway
ubuntu@l40s-180-us-west-or-1:~/qwen3_voice/Qwen3_Voice$ docker logs -f omni_voice_service
/usr/bin/python: Error while finding module specification for 'vllm.entrypoints.openai.api_server' (ModuleNotFoundError: No module named 'vllm')
/usr/bin/python: Error while finding module specification for 'vllm.entrypoints.openai.api_server' (ModuleNotFoundError: No module named 'vllm')
/usr/bin/python: Error while finding module specification for 'vllm.entrypoints.openai.api_server' (ModuleNotFoundError: No module named 'vllm')
/usr/bin/python: Error while finding module specification for 'vllm.entrypoints.openai.api_server' (ModuleNotFoundError: No module named 'vllm')
/usr/bin/python: Error while finding module specification for 'vllm.entrypoints.openai.api_server' (ModuleNotFoundError: No module named 'vllm')
/usr/bin/python: Error while finding module specification for 'vllm.entrypoints.openai.api_server' (ModuleNotFoundError: No module named 'vllm')
/usr/bin/python: Error while finding module specification for 'vllm.entrypoints.openai.api_server' (ModuleNotFoundError: No module named 'vllm')
/usr/bin/python: Error while finding module specification for 'vllm.entrypoints.openai.api_server' (ModuleNotFoundError: No module named 'vllm')
/usr/bin/python: Error while finding module specification for 'vllm.entrypoints.openai.api_server' (ModuleNotFoundError: No module named 'vllm')
/usr/bin/python: Error while finding module specification for 'vllm.entrypoints.openai.api_server' (ModuleNotFoundError: No module named 'vllm')
/usr/bin/python: Error while finding module specification for 'vllm.entrypoints.openai.api_server' (ModuleNotFoundError: No module named 'vllm')
ubuntu@l40s-180-us-west-or-1:~/qwen3_voice/Qwen3_Voice$ docker logs -f tts_read_service
/usr/local/lib/python3.12/dist-packages/transformers/utils/hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
INFO 01-31 01:29:29 [importing.py:44] Triton is installed but 0 active driver(s) found (expected 1). Disabling Triton to prevent runtime errors.
INFO 01-31 01:29:29 [importing.py:68] Triton not installed or not compatible; certain GPU-related functions will not be available.
Traceback (most recent call last):
  File "/usr/local/bin/vllm", line 4, in <module>
    from vllm.entrypoints.cli.main import main
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/__init__.py", line 4, in <module>
    from vllm.entrypoints.cli.benchmark.serve import BenchmarkServingSubcommand
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/benchmark/serve.py", line 5, in <module>
    from vllm.benchmarks.serve import add_cli_args, main
  File "/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/serve.py", line 40, in <module>
    from vllm.benchmarks.datasets import SampleRequest, add_dataset_parser, get_samples
  File "/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/datasets.py", line 38, in <module>
    from vllm.lora.utils import get_adapter_absolute_path
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/utils.py", line 22, in <module>
    from vllm.lora.layers import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/layers/__init__.py", line 14, in <module>
    from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA, FusedMoEWithLoRA
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/layers/fused_moe.py", line 18, in <module>
    from vllm.model_executor.layers.fused_moe import FusedMoE
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/__init__.py", line 11, in <module>
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/fused_moe_method_base.py", line 13, in <module>
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/modular_kernel.py", line 19, in <module>
    from vllm.model_executor.layers.fused_moe.utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/utils.py", line 9, in <module>
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/fp8_utils.py", line 22, in <module>
    from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py", line 74, in <module>
    CUTLASS_FP8_SUPPORTED = cutlass_fp8_supported()
                            ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py", line 48, in cutlass_fp8_supported
    capability_tuple = current_platform.get_device_capability()
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py", line 90, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py", line 490, in get_device_capability
    handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/third_party/pynvml.py", line 2609, in nvmlDeviceGetHandleByIndex
    _nvmlCheckReturn(ret)
  File "/usr/local/lib/python3.12/dist-packages/vllm/third_party/pynvml.py", line 1047, in _nvmlCheckReturn
    raise NVMLError(ret)
vllm.third_party.pynvml.NVMLError_InvalidArgument: Invalid Argument
/usr/local/lib/python3.12/dist-packages/transformers/utils/hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
INFO 01-31 01:29:41 [importing.py:44] Triton is installed but 0 active driver(s) found (expected 1). Disabling Triton to prevent runtime errors.
INFO 01-31 01:29:41 [importing.py:68] Triton not installed or not compatible; certain GPU-related functions will not be available.
Traceback (most recent call last):
  File "/usr/local/bin/vllm", line 4, in <module>
    from vllm.entrypoints.cli.main import main
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/__init__.py", line 4, in <module>
    from vllm.entrypoints.cli.benchmark.serve import BenchmarkServingSubcommand
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/benchmark/serve.py", line 5, in <module>
    from vllm.benchmarks.serve import add_cli_args, main
  File "/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/serve.py", line 40, in <module>
    from vllm.benchmarks.datasets import SampleRequest, add_dataset_parser, get_samples
  File "/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/datasets.py", line 38, in <module>
    from vllm.lora.utils import get_adapter_absolute_path
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/utils.py", line 22, in <module>
    from vllm.lora.layers import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/layers/__init__.py", line 14, in <module>
    from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA, FusedMoEWithLoRA
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/layers/fused_moe.py", line 18, in <module>
    from vllm.model_executor.layers.fused_moe import FusedMoE
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/__init__.py", line 11, in <module>
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/fused_moe_method_base.py", line 13, in <module>
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/modular_kernel.py", line 19, in <module>
    from vllm.model_executor.layers.fused_moe.utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/utils.py", line 9, in <module>
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/fp8_utils.py", line 22, in <module>
    from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py", line 74, in <module>
    CUTLASS_FP8_SUPPORTED = cutlass_fp8_supported()
                            ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py", line 48, in cutlass_fp8_supported
    capability_tuple = current_platform.get_device_capability()
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py", line 90, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py", line 490, in get_device_capability
    handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/third_party/pynvml.py", line 2609, in nvmlDeviceGetHandleByIndex
    _nvmlCheckReturn(ret)
  File "/usr/local/lib/python3.12/dist-packages/vllm/third_party/pynvml.py", line 1047, in _nvmlCheckReturn
    raise NVMLError(ret)
vllm.third_party.pynvml.NVMLError_InvalidArgument: Invalid Argument
/usr/local/lib/python3.12/dist-packages/transformers/utils/hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
INFO 01-31 01:29:53 [importing.py:44] Triton is installed but 0 active driver(s) found (expected 1). Disabling Triton to prevent runtime errors.
INFO 01-31 01:29:53 [importing.py:68] Triton not installed or not compatible; certain GPU-related functions will not be available.
Traceback (most recent call last):
  File "/usr/local/bin/vllm", line 4, in <module>
    from vllm.entrypoints.cli.main import main
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/__init__.py", line 4, in <module>
    from vllm.entrypoints.cli.benchmark.serve import BenchmarkServingSubcommand
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/benchmark/serve.py", line 5, in <module>
    from vllm.benchmarks.serve import add_cli_args, main
  File "/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/serve.py", line 40, in <module>
    from vllm.benchmarks.datasets import SampleRequest, add_dataset_parser, get_samples
  File "/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/datasets.py", line 38, in <module>
    from vllm.lora.utils import get_adapter_absolute_path
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/utils.py", line 22, in <module>
    from vllm.lora.layers import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/layers/__init__.py", line 14, in <module>
    from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA, FusedMoEWithLoRA
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/layers/fused_moe.py", line 18, in <module>
    from vllm.model_executor.layers.fused_moe import FusedMoE
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/__init__.py", line 11, in <module>
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/fused_moe_method_base.py", line 13, in <module>
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/modular_kernel.py", line 19, in <module>
    from vllm.model_executor.layers.fused_moe.utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/utils.py", line 9, in <module>
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/fp8_utils.py", line 22, in <module>
    from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py", line 74, in <module>
    CUTLASS_FP8_SUPPORTED = cutlass_fp8_supported()
                            ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py", line 48, in cutlass_fp8_supported
    capability_tuple = current_platform.get_device_capability()
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py", line 90, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py", line 490, in get_device_capability
    handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/third_party/pynvml.py", line 2609, in nvmlDeviceGetHandleByIndex
    _nvmlCheckReturn(ret)
  File "/usr/local/lib/python3.12/dist-packages/vllm/third_party/pynvml.py", line 1047, in _nvmlCheckReturn
    raise NVMLError(ret)
vllm.third_party.pynvml.NVMLError_InvalidArgument: Invalid Argument
/usr/local/lib/python3.12/dist-packages/transformers/utils/hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
INFO 01-31 01:30:05 [importing.py:44] Triton is installed but 0 active driver(s) found (expected 1). Disabling Triton to prevent runtime errors.
INFO 01-31 01:30:05 [importing.py:68] Triton not installed or not compatible; certain GPU-related functions will not be available.
Traceback (most recent call last):
  File "/usr/local/bin/vllm", line 4, in <module>
    from vllm.entrypoints.cli.main import main
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/__init__.py", line 4, in <module>
    from vllm.entrypoints.cli.benchmark.serve import BenchmarkServingSubcommand
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/benchmark/serve.py", line 5, in <module>
    from vllm.benchmarks.serve import add_cli_args, main
  File "/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/serve.py", line 40, in <module>
    from vllm.benchmarks.datasets import SampleRequest, add_dataset_parser, get_samples
  File "/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/datasets.py", line 38, in <module>
    from vllm.lora.utils import get_adapter_absolute_path
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/utils.py", line 22, in <module>
    from vllm.lora.layers import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/layers/__init__.py", line 14, in <module>
    from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA, FusedMoEWithLoRA
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/layers/fused_moe.py", line 18, in <module>
    from vllm.model_executor.layers.fused_moe import FusedMoE
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/__init__.py", line 11, in <module>
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/fused_moe_method_base.py", line 13, in <module>
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/modular_kernel.py", line 19, in <module>
    from vllm.model_executor.layers.fused_moe.utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/utils.py", line 9, in <module>
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/fp8_utils.py", line 22, in <module>
    from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py", line 74, in <module>
    CUTLASS_FP8_SUPPORTED = cutlass_fp8_supported()
                            ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py", line 48, in cutlass_fp8_supported
    capability_tuple = current_platform.get_device_capability()
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py", line 90, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py", line 490, in get_device_capability
    handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/third_party/pynvml.py", line 2609, in nvmlDeviceGetHandleByIndex
    _nvmlCheckReturn(ret)
  File "/usr/local/lib/python3.12/dist-packages/vllm/third_party/pynvml.py", line 1047, in _nvmlCheckReturn
    raise NVMLError(ret)
vllm.third_party.pynvml.NVMLError_InvalidArgument: Invalid Argument
/usr/local/lib/python3.12/dist-packages/transformers/utils/hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
INFO 01-31 01:30:16 [importing.py:44] Triton is installed but 0 active driver(s) found (expected 1). Disabling Triton to prevent runtime errors.
INFO 01-31 01:30:16 [importing.py:68] Triton not installed or not compatible; certain GPU-related functions will not be available.
Traceback (most recent call last):
  File "/usr/local/bin/vllm", line 4, in <module>
    from vllm.entrypoints.cli.main import main
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/__init__.py", line 4, in <module>
    from vllm.entrypoints.cli.benchmark.serve import BenchmarkServingSubcommand
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/benchmark/serve.py", line 5, in <module>
    from vllm.benchmarks.serve import add_cli_args, main
  File "/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/serve.py", line 40, in <module>
    from vllm.benchmarks.datasets import SampleRequest, add_dataset_parser, get_samples
  File "/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/datasets.py", line 38, in <module>
    from vllm.lora.utils import get_adapter_absolute_path
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/utils.py", line 22, in <module>
    from vllm.lora.layers import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/layers/__init__.py", line 14, in <module>
    from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA, FusedMoEWithLoRA
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/layers/fused_moe.py", line 18, in <module>
    from vllm.model_executor.layers.fused_moe import FusedMoE
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/__init__.py", line 11, in <module>
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/fused_moe_method_base.py", line 13, in <module>
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/modular_kernel.py", line 19, in <module>
    from vllm.model_executor.layers.fused_moe.utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/utils.py", line 9, in <module>
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/fp8_utils.py", line 22, in <module>
    from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py", line 74, in <module>
    CUTLASS_FP8_SUPPORTED = cutlass_fp8_supported()
                            ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py", line 48, in cutlass_fp8_supported
    capability_tuple = current_platform.get_device_capability()
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py", line 90, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py", line 490, in get_device_capability
    handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/third_party/pynvml.py", line 2609, in nvmlDeviceGetHandleByIndex
    _nvmlCheckReturn(ret)
  File "/usr/local/lib/python3.12/dist-packages/vllm/third_party/pynvml.py", line 1047, in _nvmlCheckReturn
    raise NVMLError(ret)
vllm.third_party.pynvml.NVMLError_InvalidArgument: Invalid Argument
/usr/local/lib/python3.12/dist-packages/transformers/utils/hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
INFO 01-31 01:30:28 [importing.py:44] Triton is installed but 0 active driver(s) found (expected 1). Disabling Triton to prevent runtime errors.
INFO 01-31 01:30:28 [importing.py:68] Triton not installed or not compatible; certain GPU-related functions will not be available.
Traceback (most recent call last):
  File "/usr/local/bin/vllm", line 4, in <module>
    from vllm.entrypoints.cli.main import main
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/__init__.py", line 4, in <module>
    from vllm.entrypoints.cli.benchmark.serve import BenchmarkServingSubcommand
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/benchmark/serve.py", line 5, in <module>
    from vllm.benchmarks.serve import add_cli_args, main
  File "/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/serve.py", line 40, in <module>
    from vllm.benchmarks.datasets import SampleRequest, add_dataset_parser, get_samples
  File "/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/datasets.py", line 38, in <module>
    from vllm.lora.utils import get_adapter_absolute_path
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/utils.py", line 22, in <module>
    from vllm.lora.layers import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/layers/__init__.py", line 14, in <module>
    from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA, FusedMoEWithLoRA
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/layers/fused_moe.py", line 18, in <module>
    from vllm.model_executor.layers.fused_moe import FusedMoE
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/__init__.py", line 11, in <module>
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/fused_moe_method_base.py", line 13, in <module>
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/modular_kernel.py", line 19, in <module>
    from vllm.model_executor.layers.fused_moe.utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/utils.py", line 9, in <module>
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/fp8_utils.py", line 22, in <module>
    from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py", line 74, in <module>
    CUTLASS_FP8_SUPPORTED = cutlass_fp8_supported()
                            ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py", line 48, in cutlass_fp8_supported
    capability_tuple = current_platform.get_device_capability()
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py", line 90, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py", line 490, in get_device_capability
    handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/third_party/pynvml.py", line 2609, in nvmlDeviceGetHandleByIndex
    _nvmlCheckReturn(ret)
  File "/usr/local/lib/python3.12/dist-packages/vllm/third_party/pynvml.py", line 1047, in _nvmlCheckReturn
    raise NVMLError(ret)
vllm.third_party.pynvml.NVMLError_InvalidArgument: Invalid Argument
/usr/local/lib/python3.12/dist-packages/transformers/utils/hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
INFO 01-31 01:30:39 [importing.py:44] Triton is installed but 0 active driver(s) found (expected 1). Disabling Triton to prevent runtime errors.
INFO 01-31 01:30:39 [importing.py:68] Triton not installed or not compatible; certain GPU-related functions will not be available.
Traceback (most recent call last):
  File "/usr/local/bin/vllm", line 4, in <module>
    from vllm.entrypoints.cli.main import main
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/__init__.py", line 4, in <module>
    from vllm.entrypoints.cli.benchmark.serve import BenchmarkServingSubcommand
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/benchmark/serve.py", line 5, in <module>
    from vllm.benchmarks.serve import add_cli_args, main
  File "/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/serve.py", line 40, in <module>
    from vllm.benchmarks.datasets import SampleRequest, add_dataset_parser, get_samples
  File "/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/datasets.py", line 38, in <module>
    from vllm.lora.utils import get_adapter_absolute_path
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/utils.py", line 22, in <module>
    from vllm.lora.layers import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/layers/__init__.py", line 14, in <module>
    from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA, FusedMoEWithLoRA
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/layers/fused_moe.py", line 18, in <module>
    from vllm.model_executor.layers.fused_moe import FusedMoE
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/__init__.py", line 11, in <module>
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/fused_moe_method_base.py", line 13, in <module>
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/modular_kernel.py", line 19, in <module>
    from vllm.model_executor.layers.fused_moe.utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/utils.py", line 9, in <module>
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/fp8_utils.py", line 22, in <module>
    from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py", line 74, in <module>
    CUTLASS_FP8_SUPPORTED = cutlass_fp8_supported()
                            ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py", line 48, in cutlass_fp8_supported
    capability_tuple = current_platform.get_device_capability()
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py", line 90, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py", line 490, in get_device_capability
    handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/third_party/pynvml.py", line 2609, in nvmlDeviceGetHandleByIndex
    _nvmlCheckReturn(ret)
  File "/usr/local/lib/python3.12/dist-packages/vllm/third_party/pynvml.py", line 1047, in _nvmlCheckReturn
    raise NVMLError(ret)
vllm.third_party.pynvml.NVMLError_InvalidArgument: Invalid Argument
/usr/local/lib/python3.12/dist-packages/transformers/utils/hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
INFO 01-31 01:30:51 [importing.py:44] Triton is installed but 0 active driver(s) found (expected 1). Disabling Triton to prevent runtime errors.
INFO 01-31 01:30:51 [importing.py:68] Triton not installed or not compatible; certain GPU-related functions will not be available.
Traceback (most recent call last):
  File "/usr/local/bin/vllm", line 4, in <module>
    from vllm.entrypoints.cli.main import main
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/__init__.py", line 4, in <module>
    from vllm.entrypoints.cli.benchmark.serve import BenchmarkServingSubcommand
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/benchmark/serve.py", line 5, in <module>
    from vllm.benchmarks.serve import add_cli_args, main
  File "/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/serve.py", line 40, in <module>
    from vllm.benchmarks.datasets import SampleRequest, add_dataset_parser, get_samples
  File "/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/datasets.py", line 38, in <module>
    from vllm.lora.utils import get_adapter_absolute_path
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/utils.py", line 22, in <module>
    from vllm.lora.layers import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/layers/__init__.py", line 14, in <module>
    from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA, FusedMoEWithLoRA
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/layers/fused_moe.py", line 18, in <module>
    from vllm.model_executor.layers.fused_moe import FusedMoE
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/__init__.py", line 11, in <module>
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/fused_moe_method_base.py", line 13, in <module>
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/modular_kernel.py", line 19, in <module>
    from vllm.model_executor.layers.fused_moe.utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/utils.py", line 9, in <module>
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/fp8_utils.py", line 22, in <module>
    from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py", line 74, in <module>
    CUTLASS_FP8_SUPPORTED = cutlass_fp8_supported()
                            ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py", line 48, in cutlass_fp8_supported
    capability_tuple = current_platform.get_device_capability()
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py", line 90, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py", line 490, in get_device_capability
    handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/third_party/pynvml.py", line 2609, in nvmlDeviceGetHandleByIndex
    _nvmlCheckReturn(ret)
  File "/usr/local/lib/python3.12/dist-packages/vllm/third_party/pynvml.py", line 1047, in _nvmlCheckReturn
    raise NVMLError(ret)
vllm.third_party.pynvml.NVMLError_InvalidArgument: Invalid Argument
/usr/local/lib/python3.12/dist-packages/transformers/utils/hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
INFO 01-31 01:31:03 [importing.py:44] Triton is installed but 0 active driver(s) found (expected 1). Disabling Triton to prevent runtime errors.
INFO 01-31 01:31:03 [importing.py:68] Triton not installed or not compatible; certain GPU-related functions will not be available.
Traceback (most recent call last):
  File "/usr/local/bin/vllm", line 4, in <module>
    from vllm.entrypoints.cli.main import main
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/__init__.py", line 4, in <module>
    from vllm.entrypoints.cli.benchmark.serve import BenchmarkServingSubcommand
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/benchmark/serve.py", line 5, in <module>
    from vllm.benchmarks.serve import add_cli_args, main
  File "/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/serve.py", line 40, in <module>
    from vllm.benchmarks.datasets import SampleRequest, add_dataset_parser, get_samples
  File "/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/datasets.py", line 38, in <module>
    from vllm.lora.utils import get_adapter_absolute_path
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/utils.py", line 22, in <module>
    from vllm.lora.layers import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/layers/__init__.py", line 14, in <module>
    from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA, FusedMoEWithLoRA
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/layers/fused_moe.py", line 18, in <module>
    from vllm.model_executor.layers.fused_moe import FusedMoE
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/__init__.py", line 11, in <module>
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/fused_moe_method_base.py", line 13, in <module>
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/modular_kernel.py", line 19, in <module>
    from vllm.model_executor.layers.fused_moe.utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/utils.py", line 9, in <module>
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/fp8_utils.py", line 22, in <module>
    from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py", line 74, in <module>
    CUTLASS_FP8_SUPPORTED = cutlass_fp8_supported()
                            ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py", line 48, in cutlass_fp8_supported
    capability_tuple = current_platform.get_device_capability()
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py", line 90, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py", line 490, in get_device_capability
    handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/third_party/pynvml.py", line 2609, in nvmlDeviceGetHandleByIndex
    _nvmlCheckReturn(ret)
  File "/usr/local/lib/python3.12/dist-packages/vllm/third_party/pynvml.py", line 1047, in _nvmlCheckReturn
    raise NVMLError(ret)
vllm.third_party.pynvml.NVMLError_InvalidArgument: Invalid Argument
/usr/local/lib/python3.12/dist-packages/transformers/utils/hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
INFO 01-31 01:31:13 [importing.py:44] Triton is installed but 0 active driver(s) found (expected 1). Disabling Triton to prevent runtime errors.
INFO 01-31 01:31:13 [importing.py:68] Triton not installed or not compatible; certain GPU-related functions will not be available.
Traceback (most recent call last):
  File "/usr/local/bin/vllm", line 4, in <module>
    from vllm.entrypoints.cli.main import main
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/__init__.py", line 4, in <module>
    from vllm.entrypoints.cli.benchmark.serve import BenchmarkServingSubcommand
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/benchmark/serve.py", line 5, in <module>
    from vllm.benchmarks.serve import add_cli_args, main
  File "/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/serve.py", line 40, in <module>
    from vllm.benchmarks.datasets import SampleRequest, add_dataset_parser, get_samples
  File "/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/datasets.py", line 38, in <module>
    from vllm.lora.utils import get_adapter_absolute_path
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/utils.py", line 22, in <module>
    from vllm.lora.layers import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/layers/__init__.py", line 14, in <module>
    from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA, FusedMoEWithLoRA
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/layers/fused_moe.py", line 18, in <module>
    from vllm.model_executor.layers.fused_moe import FusedMoE
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/__init__.py", line 11, in <module>
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/fused_moe_method_base.py", line 13, in <module>
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/modular_kernel.py", line 19, in <module>
    from vllm.model_executor.layers.fused_moe.utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/utils.py", line 9, in <module>
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/fp8_utils.py", line 22, in <module>
    from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py", line 74, in <module>
    CUTLASS_FP8_SUPPORTED = cutlass_fp8_supported()
                            ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py", line 48, in cutlass_fp8_supported
    capability_tuple = current_platform.get_device_capability()
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py", line 90, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py", line 490, in get_device_capability
    handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/third_party/pynvml.py", line 2609, in nvmlDeviceGetHandleByIndex
    _nvmlCheckReturn(ret)
  File "/usr/local/lib/python3.12/dist-packages/vllm/third_party/pynvml.py", line 1047, in _nvmlCheckReturn
    raise NVMLError(ret)
vllm.third_party.pynvml.NVMLError_InvalidArgument: Invalid Argument
/usr/local/lib/python3.12/dist-packages/transformers/utils/hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
INFO 01-31 01:31:25 [importing.py:44] Triton is installed but 0 active driver(s) found (expected 1). Disabling Triton to prevent runtime errors.
INFO 01-31 01:31:25 [importing.py:68] Triton not installed or not compatible; certain GPU-related functions will not be available.
Traceback (most recent call last):
  File "/usr/local/bin/vllm", line 4, in <module>
    from vllm.entrypoints.cli.main import main
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/__init__.py", line 4, in <module>
    from vllm.entrypoints.cli.benchmark.serve import BenchmarkServingSubcommand
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/benchmark/serve.py", line 5, in <module>
    from vllm.benchmarks.serve import add_cli_args, main
  File "/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/serve.py", line 40, in <module>
    from vllm.benchmarks.datasets import SampleRequest, add_dataset_parser, get_samples
  File "/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/datasets.py", line 38, in <module>
    from vllm.lora.utils import get_adapter_absolute_path
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/utils.py", line 22, in <module>
    from vllm.lora.layers import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/layers/__init__.py", line 14, in <module>
    from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA, FusedMoEWithLoRA
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/layers/fused_moe.py", line 18, in <module>
    from vllm.model_executor.layers.fused_moe import FusedMoE
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/__init__.py", line 11, in <module>
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/fused_moe_method_base.py", line 13, in <module>
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/modular_kernel.py", line 19, in <module>
    from vllm.model_executor.layers.fused_moe.utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/utils.py", line 9, in <module>
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/fp8_utils.py", line 22, in <module>
    from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py", line 74, in <module>
    CUTLASS_FP8_SUPPORTED = cutlass_fp8_supported()
                            ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py", line 48, in cutlass_fp8_supported
    capability_tuple = current_platform.get_device_capability()
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py", line 90, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py", line 490, in get_device_capability
    handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/third_party/pynvml.py", line 2609, in nvmlDeviceGetHandleByIndex
    _nvmlCheckReturn(ret)
  File "/usr/local/lib/python3.12/dist-packages/vllm/third_party/pynvml.py", line 1047, in _nvmlCheckReturn
    raise NVMLError(ret)
vllm.third_party.pynvml.NVMLError_InvalidArgument: Invalid Argument
/usr/local/lib/python3.12/dist-packages/transformers/utils/hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
INFO 01-31 01:31:36 [importing.py:44] Triton is installed but 0 active driver(s) found (expected 1). Disabling Triton to prevent runtime errors.
INFO 01-31 01:31:36 [importing.py:68] Triton not installed or not compatible; certain GPU-related functions will not be available.
Traceback (most recent call last):
  File "/usr/local/bin/vllm", line 4, in <module>
    from vllm.entrypoints.cli.main import main
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/__init__.py", line 4, in <module>
    from vllm.entrypoints.cli.benchmark.serve import BenchmarkServingSubcommand
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/benchmark/serve.py", line 5, in <module>
    from vllm.benchmarks.serve import add_cli_args, main
  File "/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/serve.py", line 40, in <module>
    from vllm.benchmarks.datasets import SampleRequest, add_dataset_parser, get_samples
  File "/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/datasets.py", line 38, in <module>
    from vllm.lora.utils import get_adapter_absolute_path
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/utils.py", line 22, in <module>
    from vllm.lora.layers import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/layers/__init__.py", line 14, in <module>
    from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA, FusedMoEWithLoRA
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/layers/fused_moe.py", line 18, in <module>
    from vllm.model_executor.layers.fused_moe import FusedMoE
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/__init__.py", line 11, in <module>
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/fused_moe_method_base.py", line 13, in <module>
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/modular_kernel.py", line 19, in <module>
    from vllm.model_executor.layers.fused_moe.utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/utils.py", line 9, in <module>
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/fp8_utils.py", line 22, in <module>
    from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py", line 74, in <module>
    CUTLASS_FP8_SUPPORTED = cutlass_fp8_supported()
                            ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py", line 48, in cutlass_fp8_supported
    capability_tuple = current_platform.get_device_capability()
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py", line 90, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py", line 490, in get_device_capability
    handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/third_party/pynvml.py", line 2609, in nvmlDeviceGetHandleByIndex
    _nvmlCheckReturn(ret)
  File "/usr/local/lib/python3.12/dist-packages/vllm/third_party/pynvml.py", line 1047, in _nvmlCheckReturn
    raise NVMLError(ret)
vllm.third_party.pynvml.NVMLError_InvalidArgument: Invalid Argument
/usr/local/lib/python3.12/dist-packages/transformers/utils/hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
INFO 01-31 01:31:48 [importing.py:44] Triton is installed but 0 active driver(s) found (expected 1). Disabling Triton to prevent runtime errors.
INFO 01-31 01:31:48 [importing.py:68] Triton not installed or not compatible; certain GPU-related functions will not be available.
Traceback (most recent call last):
  File "/usr/local/bin/vllm", line 4, in <module>
    from vllm.entrypoints.cli.main import main
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/__init__.py", line 4, in <module>
    from vllm.entrypoints.cli.benchmark.serve import BenchmarkServingSubcommand
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/benchmark/serve.py", line 5, in <module>
    from vllm.benchmarks.serve import add_cli_args, main
  File "/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/serve.py", line 40, in <module>
    from vllm.benchmarks.datasets import SampleRequest, add_dataset_parser, get_samples
  File "/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/datasets.py", line 38, in <module>
    from vllm.lora.utils import get_adapter_absolute_path
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/utils.py", line 22, in <module>
    from vllm.lora.layers import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/layers/__init__.py", line 14, in <module>
    from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA, FusedMoEWithLoRA
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/layers/fused_moe.py", line 18, in <module>
    from vllm.model_executor.layers.fused_moe import FusedMoE
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/__init__.py", line 11, in <module>
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/fused_moe_method_base.py", line 13, in <module>
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/modular_kernel.py", line 19, in <module>
    from vllm.model_executor.layers.fused_moe.utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/utils.py", line 9, in <module>
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/fp8_utils.py", line 22, in <module>
    from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py", line 74, in <module>
    CUTLASS_FP8_SUPPORTED = cutlass_fp8_supported()
                            ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py", line 48, in cutlass_fp8_supported
    capability_tuple = current_platform.get_device_capability()
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py", line 90, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py", line 490, in get_device_capability
    handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/third_party/pynvml.py", line 2609, in nvmlDeviceGetHandleByIndex
    _nvmlCheckReturn(ret)
  File "/usr/local/lib/python3.12/dist-packages/vllm/third_party/pynvml.py", line 1047, in _nvmlCheckReturn
    raise NVMLError(ret)
vllm.third_party.pynvml.NVMLError_InvalidArgument: Invalid Argument
/usr/local/lib/python3.12/dist-packages/transformers/utils/hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
INFO 01-31 01:31:59 [importing.py:44] Triton is installed but 0 active driver(s) found (expected 1). Disabling Triton to prevent runtime errors.
INFO 01-31 01:31:59 [importing.py:68] Triton not installed or not compatible; certain GPU-related functions will not be available.
Traceback (most recent call last):
  File "/usr/local/bin/vllm", line 4, in <module>
    from vllm.entrypoints.cli.main import main
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/__init__.py", line 4, in <module>
    from vllm.entrypoints.cli.benchmark.serve import BenchmarkServingSubcommand
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/benchmark/serve.py", line 5, in <module>
    from vllm.benchmarks.serve import add_cli_args, main
  File "/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/serve.py", line 40, in <module>
    from vllm.benchmarks.datasets import SampleRequest, add_dataset_parser, get_samples
  File "/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/datasets.py", line 38, in <module>
    from vllm.lora.utils import get_adapter_absolute_path
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/utils.py", line 22, in <module>
    from vllm.lora.layers import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/layers/__init__.py", line 14, in <module>
    from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA, FusedMoEWithLoRA
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/layers/fused_moe.py", line 18, in <module>
    from vllm.model_executor.layers.fused_moe import FusedMoE
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/__init__.py", line 11, in <module>
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/fused_moe_method_base.py", line 13, in <module>
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/modular_kernel.py", line 19, in <module>
    from vllm.model_executor.layers.fused_moe.utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/utils.py", line 9, in <module>
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/fp8_utils.py", line 22, in <module>
    from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py", line 74, in <module>
    CUTLASS_FP8_SUPPORTED = cutlass_fp8_supported()
                            ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py", line 48, in cutlass_fp8_supported
    capability_tuple = current_platform.get_device_capability()
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py", line 90, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py", line 490, in get_device_capability
    handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/third_party/pynvml.py", line 2609, in nvmlDeviceGetHandleByIndex
    _nvmlCheckReturn(ret)
  File "/usr/local/lib/python3.12/dist-packages/vllm/third_party/pynvml.py", line 1047, in _nvmlCheckReturn
    raise NVMLError(ret)
vllm.third_party.pynvml.NVMLError_InvalidArgument: Invalid Argument
/usr/local/lib/python3.12/dist-packages/transformers/utils/hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
INFO 01-31 01:32:11 [importing.py:44] Triton is installed but 0 active driver(s) found (expected 1). Disabling Triton to prevent runtime errors.
INFO 01-31 01:32:11 [importing.py:68] Triton not installed or not compatible; certain GPU-related functions will not be available.
Traceback (most recent call last):
  File "/usr/local/bin/vllm", line 4, in <module>
    from vllm.entrypoints.cli.main import main
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/__init__.py", line 4, in <module>
    from vllm.entrypoints.cli.benchmark.serve import BenchmarkServingSubcommand
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/benchmark/serve.py", line 5, in <module>
    from vllm.benchmarks.serve import add_cli_args, main
  File "/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/serve.py", line 40, in <module>
    from vllm.benchmarks.datasets import SampleRequest, add_dataset_parser, get_samples
  File "/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/datasets.py", line 38, in <module>
    from vllm.lora.utils import get_adapter_absolute_path
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/utils.py", line 22, in <module>
    from vllm.lora.layers import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/layers/__init__.py", line 14, in <module>
    from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA, FusedMoEWithLoRA
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/layers/fused_moe.py", line 18, in <module>
    from vllm.model_executor.layers.fused_moe import FusedMoE
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/__init__.py", line 11, in <module>
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/fused_moe_method_base.py", line 13, in <module>
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/modular_kernel.py", line 19, in <module>
    from vllm.model_executor.layers.fused_moe.utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/utils.py", line 9, in <module>
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/fp8_utils.py", line 22, in <module>
    from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py", line 74, in <module>
    CUTLASS_FP8_SUPPORTED = cutlass_fp8_supported()
                            ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py", line 48, in cutlass_fp8_supported
    capability_tuple = current_platform.get_device_capability()
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py", line 90, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py", line 490, in get_device_capability
    handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/third_party/pynvml.py", line 2609, in nvmlDeviceGetHandleByIndex
    _nvmlCheckReturn(ret)
  File "/usr/local/lib/python3.12/dist-packages/vllm/third_party/pynvml.py", line 1047, in _nvmlCheckReturn
    raise NVMLError(ret)
vllm.third_party.pynvml.NVMLError_InvalidArgument: Invalid Argument
/usr/local/lib/python3.12/dist-packages/transformers/utils/hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
INFO 01-31 01:32:23 [importing.py:44] Triton is installed but 0 active driver(s) found (expected 1). Disabling Triton to prevent runtime errors.
INFO 01-31 01:32:23 [importing.py:68] Triton not installed or not compatible; certain GPU-related functions will not be available.
Traceback (most recent call last):
  File "/usr/local/bin/vllm", line 4, in <module>
    from vllm.entrypoints.cli.main import main
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/__init__.py", line 4, in <module>
    from vllm.entrypoints.cli.benchmark.serve import BenchmarkServingSubcommand
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/benchmark/serve.py", line 5, in <module>
    from vllm.benchmarks.serve import add_cli_args, main
  File "/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/serve.py", line 40, in <module>
    from vllm.benchmarks.datasets import SampleRequest, add_dataset_parser, get_samples
  File "/usr/local/lib/python3.12/dist-packages/vllm/benchmarks/datasets.py", line 38, in <module>
    from vllm.lora.utils import get_adapter_absolute_path
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/utils.py", line 22, in <module>
    from vllm.lora.layers import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/layers/__init__.py", line 14, in <module>
    from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA, FusedMoEWithLoRA
  File "/usr/local/lib/python3.12/dist-packages/vllm/lora/layers/fused_moe.py", line 18, in <module>
    from vllm.model_executor.layers.fused_moe import FusedMoE
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/__init__.py", line 11, in <module>
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/fused_moe_method_base.py", line 13, in <module>
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/modular_kernel.py", line 19, in <module>
    from vllm.model_executor.layers.fused_moe.utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/utils.py", line 9, in <module>
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/fp8_utils.py", line 22, in <module>
    from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py", line 74, in <module>
    CUTLASS_FP8_SUPPORTED = cutlass_fp8_supported()
                            ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py", line 48, in cutlass_fp8_supported
    capability_tuple = current_platform.get_device_capability()
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py", line 90, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py", line 490, in get_device_capability
    handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/third_party/pynvml.py", line 2609, in nvmlDeviceGetHandleByIndex
    _nvmlCheckReturn(ret)
  File "/usr/local/lib/python3.12/dist-packages/vllm/third_party/pynvml.py", line 1047, in _nvmlCheckReturn
    raise NVMLError(ret)
vllm.third_party.pynvml.NVMLError_InvalidArgument: Invalid Argument
ubuntu@l40s-180-us-west-or-1:~/qwen3_voice/Qwen3_Voice$ docker ps
CONTAINER ID   IMAGE                                 COMMAND                  CREATED         STATUS                                  PORTS                                         NAMES
10134a8b674d   vllm/vllm-openai:latest               "vllm serve python -…"   3 minutes ago   Restarting (1) Less than a second ago                                                 tts_read_service
cb0bb01cb3c2   qwen3_voice_omni-service              "python -m vllm.entr…"   3 minutes ago   Restarting (1) 50 seconds ago                                                         omni_voice_service
8dbf15eb35e1   ghcr.io/berriai/litellm:main-latest   "docker/prod_entrypo…"   7 days ago      Up 7 days                               0.0.0.0:4000->4000/tcp, [::]:4000->4000/tcp   aether-gateway
5617c613f933   redis/redis-stack-server:latest       "/entrypoint.sh"         7 days ago      Up 7 days                               0.0.0.0:6390->6379/tcp, [::]:6390->6379/tcp   litellm-redis
11b0aa777c7a   vllm/vllm-openai:latest               "vllm serve --model …"   7 days ago      Up 7 days                               0.0.0.0:8001->8001/tcp, [::]:8001->8001/tcp   qwen3-vl-thinking
ubuntu@l40s-180-us-west-or-1:~/qwen3_voice/Qwen3_Voice$ 
