You are a senior infrastructure engineer.

I am replacing a legacy audio stack with a modern Omni-based architecture on Ubuntu 22.04.

Hardware:
- NVIDIA L40S (48GB VRAM)
- GPU index to use: device=1

Goal:
Deploy a Dockerized, production-ready voice service using vLLM Omni that supports:
- Real-time audio-to-audio conversation
- A separate lightweight read-aloud TTS endpoint

Models:

1) Voice-to-Voice (primary service)
   - Model: cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit
   - Engine: vLLM-Omni (Thinker–Talker architecture)
   - Interface: WebSocket audio streaming
   - Port mapping: 8004 (host) → 8000 (container)

2) Read-Aloud (secondary service)
   - Model: Qwen/Qwen3-TTS-12Hz-0.6B-Base
   - Interface: HTTP POST
   - Port mapping: 8005 (host) → 8001 (container)

Requirements:

- Use the official vLLM-Omni Docker image for the Omni model.
- Enable the --omni flag and streaming audio output.
- GPU pinning: device=1 for both containers.
- VRAM budgets:
    - Omni: --gpu-memory-utilization 0.65
    - TTS:  --gpu-memory-utilization 0.10
- Audio output: 24kHz PCM
- Both containers must be on the same Docker bridge network.

FastAPI Orchestration Layer:

- Provide a Python FastAPI app (main.py) that exposes:
    /voice (WebSocket):
        - Accepts raw PCM audio
        - Streams audio to the Omni model
        - Returns the Talker audio stream immediately (audio-to-audio)

    /read (POST):
        - Accepts text
        - Forwards to the 0.6B TTS model
        - Returns the generated audio file for UI "read aloud"

Deliverables:
- docker-compose.yml
- Dockerfile (if needed)
- main.py (FastAPI router + WebSocket logic)

Assume this will replace an existing service, so ports must match exactly.
Do not include unrelated explanations. Produce runnable code.

## 🔧 **APPEND: Required API Endpoints & Behavioral Contracts**

### Objective

Extend the service into a **production-grade, introspectable, multi-modal voice gateway** by implementing the following HTTP and WebSocket endpoints.
All endpoints must be lightweight, stateless where possible, and safe for orchestration environments (Docker, Compose, LiteLLM, fleet managers).

---

### 1. Core Operational Endpoints (Mandatory)

#### `GET /health`

**Purpose:** Service liveness + readiness check.

**Must return:**

* `status`: `"ok" | "loading" | "error"`
* `model_loaded`: boolean
* `active_models`: list of model identifiers
* `gpu_id`: integer or `"cpu"`
* `vram_used_mb`
* `vram_total_mb`
* `uptime_seconds`

This endpoint must return **200 OK** only when the service is fully ready to accept requests.

---

#### `GET /models`

**Purpose:** Runtime introspection of loaded models.

**Must return (per model):**

* `name`
* `role` (e.g. `"omni"`, `"tts"`)
* `modalities` (e.g. `["text","audio","vision"]`)
* `quantization`
* `max_context`
* `audio_capabilities` (tts / stt / full duplex)

This endpoint is required for future **model fleet management**.

---

#### `GET /config`

**Purpose:** Debug + audit visibility.

**Must include:**

* audio sample rate
* streaming mode enabled/disabled
* default language
* container port mappings
* environment flags relevant to inference

This endpoint must be **read-only**.

---

### 2. Voice & Audio Endpoints (Primary Functionality)

#### `WS /voice`

**Purpose:** Full-duplex real-time voice interaction (audio in → audio out).

**Behavior:**

* Accepts streamed audio input
* Emits streamed synthesized audio output
* Supports interruption / cancel mid-stream
* Designed for conversational use

---

#### `WS /voice/stream`

**Purpose:** Semantic alias of `/voice`.

This endpoint must behave identically to `/voice` but exists to clearly express **intent** for clients and future routing logic.

---

#### `POST /voice/batch`

**Purpose:** Non-streaming TTS generation.

**Input:**

* text
* optional voice / language parameters

**Output:**

* complete audio file (WAV or PCM)

Used for offline generation and file-based workflows.

---

### 3. Read-Aloud / Accessibility Endpoints

#### `POST /read`

**Purpose:** Convert long text into spoken audio (single response).

**Behavior:**

* Optimized for long-form reading
* Stable pacing
* Minimal latency spikes

---

#### `WS /read/stream`

**Purpose:** Streamed read-aloud.

**Behavior:**

* Emits audio chunks incrementally
* Supports pause / resume / cancel
* Designed for UI “Read Aloud” buttons

---

### 4. Language & Transcription Endpoints

#### `GET /languages`

**Purpose:** Capability discovery.

**Returns:**

* supported input languages
* supported output languages
* default language
* auto-detection supported (boolean)

---

#### `POST /voice/transcribe`

**Purpose:** Audio → text transcription.

**Behavior:**

* Accepts audio file or stream
* Returns transcription + confidence
* Used for logging, RAG ingestion, debugging

---

### 5. Metrics & Observability

#### `GET /metrics`

**Purpose:** Monitoring & performance tracking.

Expose Prometheus-style metrics including:

* request latency
* audio generation latency
* tokens/sec
* active streams
* GPU utilization

This endpoint must not block inference.

---

### 6. Design Constraints

* All endpoints must be **container-safe**
* No persistent user state inside the service
* No hard-coded model paths
* Configuration via environment variables
* Ports must remain compatible with:

  * internal: `8000`
  * external: `8004`

---

### Final Requirement

The resulting service must function as a **general-purpose audio + multimodal inference gateway**, suitable for:

* internal tooling
* desktop assistants
* client deployments
* future fleet-managed orchestration

---

