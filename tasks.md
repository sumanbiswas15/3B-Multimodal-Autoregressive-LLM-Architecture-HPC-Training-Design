# Implementation Plan: llm-3b-build

## Overview

This plan converts the five-phase design (Architecture → Data Pipeline → Pre-Training → Alignment → Safety & Deployment) into discrete, incremental coding tasks for a Python/PyTorch implementation targeting the **Param Ganga HPC cluster at IIT Roorkee** (40 × NVIDIA V100 SXM2 32 GB GPUs). The model is approximately **3 billion parameters** with a **32,768-token context window**, trained across 40 V100 GPUs using TP=2, PP=4, DP=5. Each task builds on earlier ones, ending with full integration. All 15 correctness properties from the design have corresponding property-based test sub-tasks using Hypothesis.

---

## Tasks

- [ ] 1. Project scaffold, configuration dataclasses, and smoke tests
  - Create the top-level package structure: `llm10b/`, `llm10b/model/`, `llm10b/data/`, `llm10b/train/`, `llm10b/alignment/`, `llm10b/safety/`, `llm10b/api/`, `tests/`
  - Implement all configuration dataclasses: `ModelConfig`, `MoEConfig`, `VisionConfig`, `TokenizerConfig`, `TrainingConfig`, `DataPipelineConfig`, `Checkpoint`, `APIRequest`, `APIResponse`, `SafetyClassifierResult`
  - Enforce all invariants in `__post_init__`:
    - `ModelConfig`: `use_weight_tying must be False`, `attention_backend in ("xformers", "flash_attn_v1")` (not "flash_attn_v2" — unsupported on V100 CC 7.0), `use_gradient_checkpointing must be True`
    - `MoEConfig`: `num_experts >= 64`, `1 <= top_k <= num_experts`, `dropout == 0.0`
    - `VisionConfig`: `patch_size == 16`, `max_patches == 256`
    - `TokenizerConfig`: `32_000 <= vocab_size <= 256_000`
    - `TrainingConfig`: `global_batch_tokens` in [1_000_000, 4_000_000], `use_fp16 == True`, `use_loss_scaling == True`, `tensor_parallel == 2`, `pipeline_parallel == 4`, `data_parallel == 5`, `num_gpus == 40`
  - Write smoke/config tests that assert every invariant raises `ValueError` when violated
  - Set up `pytest` with `hypothesis` installed; add `conftest.py` with shared Hypothesis profiles (`@settings(max_examples=100)`)
  - _Requirements: 1.1–1.10, 2.1–2.6, 4.1–4.7, 5.1–5.5, 8.1–8.8, 9.1–9.7, 10.1–10.5_

  - [ ] 1.1 Implement configuration dataclasses with validation
    - Implement all dataclasses listed above with `__post_init__` guards
    - `ModelConfig` must include `attention_backend: str` ("xformers" | "flash_attn_v1") and `use_gradient_checkpointing: bool` (must be True)
    - `TrainingConfig` must include `use_fp16: bool` (must be True) and `use_loss_scaling: bool` (must be True)
    - Enforce `num_experts >= 64`, `max_patches == 256`, `global_batch_tokens` in [1_000_000, 4_000_000]
    - _Requirements: 1.1, 1.2, 1.3, 1.6, 1.8, 1.9, 1.10, 2.1, 2.2, 2.4, 4.1, 4.3, 5.4, 9.7_

  - [ ]* 1.2 Write smoke/configuration tests
    - Test every `ValueError` guard in every dataclass
    - Test boundary values: `num_layers=28`, `num_layers=48`, `vocab_size=32000`, `num_experts=64`, `max_patches=256`, `global_batch_tokens=2_000_000`
    - Assert `attention_backend="flash_attn_v2"` raises `ValueError`
    - Assert `use_gradient_checkpointing=False` raises `ValueError`
    - Assert `use_fp16=False` raises `ValueError`
    - _Requirements: 1.1–1.10, 2.1, 2.4, 4.1, 5.4, 8.2, 9.1–9.7, 10.3–10.4_

- [ ] 2. Shared subword tokenizer
  - Implement `Tokenizer` class with `encode(text: str) -> List[int]` and `decode(ids: List[int]) -> str`
  - Support BPE and Unigram algorithms (wrap SentencePiece or HuggingFace tokenizers)
  - Add 256 reserved byte-level fallback tokens (0x00–0xFF)
  - Register all special tokens: `<BOS>`, `<EOS>`, `<PAD>`, `<IMG_START>`, `<IMG_END>`
  - Implement training script `train_tokenizer.py` using corpus sample (reference implementation; actual training on 100B+ tokens requires cluster)
  - _Requirements: 5.1–5.5_

  - [ ] 2.1 Implement Tokenizer class with encode/decode and byte-level fallback
    - _Requirements: 5.1, 5.2, 5.3, 5.5_

  - [ ]* 2.2 Write property test P9: Tokenizer round-trip fidelity
    - **Property 9: Tokenizer Round-Trip Fidelity**
    - **Validates: Requirements 5.2, 5.3**
    - Use `st.text()` with full Unicode strategy; exclude special control tokens
    - `@settings(max_examples=100)`

  - [ ]* 2.3 Write property test P10: Byte-level fallback completeness
    - **Property 10: Byte-Level Fallback Completeness**
    - **Validates: Requirements 5.5**
    - Use `st.binary()` for arbitrary byte sequences; assert no error raised
    - `@settings(max_examples=100)`

- [ ] 3. Transformer decoder backbone
  - Implement `TransformerDecoder` module: embedding layer, 32 decoder layers, LM head (`nn.Linear`, no weight tying)
  - Each decoder layer: `CausalSelfAttention` + `FFNLayer` (SwiGLU or GELU, configurable); hidden_dim=2,560; 20 attention heads
  - `CausalSelfAttention`: multi-head attention with causal mask, RoPE or ALiBi positional encoding (configurable via `ModelConfig`)
  - **Attention backend**: Use **xformers memory-efficient attention** (`xformers.ops.memory_efficient_attention`) or **FlashAttention-1** (`flash_attn` v1 API) — selected by `ModelConfig.attention_backend`. **Do NOT use FlashAttention-2**: it requires CUDA CC 8.0+ (Ampere) and is unsupported on V100 (CC 7.0).
  - **Gradient checkpointing**: Apply `torch.utils.checkpoint.checkpoint()` to every decoder layer (`use_gradient_checkpointing=True` is mandatory). This is required to fit 32K activations within 32 GB V100 HBM2.
  - **Mixed precision**: Train under `torch.cuda.amp.autocast(dtype=torch.float16)` with dynamic loss scaling via `torch.cuda.amp.GradScaler`. **Do NOT use BF16** — unavailable on V100 CC 7.0.
  - Final `nn.Linear` LM head → softmax over vocabulary; output logits `[B, T, V]`
  - No weight tying between embedding and LM head
  - _Requirements: 1.1–1.10_

  - [ ] 3.1 Implement RoPE and ALiBi positional encodings
    - Write `apply_rope(q, k, positions)` and `apply_alibi_bias(scores, seq_len)` utilities
    - Standard RoPE scaling is sufficient for the 32,768-token training context (no YaRN needed)
    - _Requirements: 1.5_

  - [ ] 3.2 Implement `CausalSelfAttention` with xformers/FlashAttention-1 and causal masking
    - Integrate `xformers.ops.memory_efficient_attention` or `flash_attn` v1 kernel based on `attention_backend` config
    - Expose `return_attn_weights` flag for testing (falls back to standard attention when enabled)
    - Do NOT integrate FlashAttention-2
    - _Requirements: 1.4, 3.6_

  - [ ]* 3.3 Write property test P5: Strict causal masking
    - **Property 5: Strict Causal Masking**
    - **Validates: Requirements 3.6**
    - Use random token sequences 1–512 tokens; inspect attention weight matrices with `return_attn_weights=True`
    - Assert every weight at position (i, j) where j > i is exactly 0.0
    - `@settings(max_examples=100)`

  - [ ] 3.4 Implement SwiGLU and GELU FFN layers
    - _Requirements: 1.3, 1.10_

  - [ ] 3.5 Implement full `TransformerDecoder` forward pass with gradient checkpointing
    - Stack 32 layers with `torch.utils.checkpoint.checkpoint()` wrapping each layer
    - Apply FP16 autocast; `forward(input_ids: Tensor[B, T]) -> logits: Tensor[B, T, V]`
    - _Requirements: 1.1–1.10_

  - [ ]* 3.6 Write property test P1: Output probability distribution validity
    - **Property 1: Output Probability Distribution Validity**
    - **Validates: Requirements 1.7**
    - `st.lists(st.integers(0, vocab_size-1), min_size=1, max_size=512)` for token IDs
    - Assert all logits after softmax are in [0.0, 1.0] and sum to 1.0 ± 1e-5
    - `@settings(max_examples=100)`

  - [ ]* 3.7 Write unit test: forward pass with 32,768 tokens
    - Assert valid output + no OOM on V100 target hardware (32 GB HBM2, gradient checkpointing enabled)
    - _Requirements: 3.1, 3.4_

  - [ ]* 3.8 Write unit test: reject input exceeding 32,768 tokens
    - Submit 32,769 tokens; assert structured error `{"error": "sequence_too_long", "max_tokens": 32768, "provided_tokens": 32769}`
    - _Requirements: 3.5_

- [ ] 4. Mixture-of-Experts FFN layers
  - Implement `MoELayer` with **64 experts** (each a SwiGLU FFN), learned router (`nn.Linear`, hidden_dim → 64), top-k=2 selection, softmax-over-top-k combination weights
  - Implement `compute_aux_loss(routing_logits, top_k_indices, alpha)` for load-balancing loss L_aux = α × Σ_i (f_i × P_i)
  - MoE dropout = 0.0 (enforced via `MoEConfig`)
  - Integrate `MoELayer` into `TransformerDecoder` at configurable alternating layers
  - _Requirements: 2.1–2.6_

  - [ ] 4.1 Implement `MoELayer` with router, top-k=2 selection, and weighted combination (64 experts)
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [ ]* 4.2 Write property test P2: MoE top-k routing exactness
    - **Property 2: MoE Top-k Routing Exactness**
    - **Validates: Requirements 2.2**
    - Random `[B, T, D]` tensors; inspect routing decisions; assert exactly k=2 experts activated per token
    - `@settings(max_examples=100)`

  - [ ]* 4.3 Write property test P3: MoE weighted combination correctness
    - **Property 3: MoE Weighted Combination Correctness**
    - **Validates: Requirements 2.3**
    - Random `[B, T, D]` tensors; compare `MoELayer.forward()` output to manually computed weighted sum of top-2 expert outputs
    - `@settings(max_examples=100)`

  - [ ] 4.4 Implement `compute_aux_loss` (load-balancing auxiliary loss)
    - `compute_aux_loss(f_fractions, p_probs, alpha) -> scalar`; uniform fraction baseline is 1/64
    - _Requirements: 2.5, 2.6_

  - [ ]* 4.5 Write property test P4: Load balancing loss activation
    - **Property 4: Load Balancing Loss Activation**
    - **Validates: Requirements 2.5**
    - Random routing distributions across 64 experts; vary imbalance above/below tolerance threshold
    - Assert L_aux > 0 when any expert deviates from 1/64 by more than tolerance; ≈ 0 when all within tolerance
    - `@settings(max_examples=100)`

- [ ] 5. Long-context attention mechanisms (32K tokens)
  - Implement `SlidingWindowAttention` (local window of size 4,096) using xformers or FlashAttention-1 backend
  - Implement full `GlobalAttention` for every 4th layer (layers 4, 8, 12, …, 28) using xformers memory-efficient attention — this replaces the original ChunkedGlobalAttention and is simpler at 32K scale
  - **No HierarchicalAttentionCompressor needed**: O(n²) at 32K with xformers is manageable; hierarchical compression is unnecessary complexity at this scale
  - **No YaRN/NTK-aware RoPE needed**: 32K is the training context length, not an extrapolation target; standard RoPE generalizes without interpolation tricks
  - Integrate into `TransformerDecoder`: sliding window for all non-global layers; full xformers attention for every 4th layer (layers divisible by 4)
  - All attention mechanisms must enforce causal masking and support gradient checkpointing
  - _Requirements: 3.1–3.6_

  - [ ] 5.1 Implement `SlidingWindowAttention` (window=4,096) and `GlobalAttention` (xformers, every 4th layer)
    - Both must enforce causal masking; reuse `return_attn_weights` flag for testing
    - _Requirements: 3.6_

  - [ ] 5.2 Integrate hybrid attention into `TransformerDecoder` (sliding window layers + global every 4th)
    - Layer assignment: layers 4, 8, 12, 16, 20, 24, 28 use `GlobalAttention`; all other layers use `SlidingWindowAttention`
    - _Requirements: 3.1, 3.2, 3.3_

  - [ ]* 5.3 Write integration test: long-context retrieval benchmark (32K vs 4K context)
    - Accuracy on 32K-context queries ≥ accuracy on 4K-context equivalent queries containing the same fact
    - _Requirements: 3.2_

  - [ ]* 5.4 Write integration test: cross-context coherence benchmark
    - Reason over information in both first 10% and last 10% of a 32,768-token input; evaluate consistency
    - _Requirements: 3.3_

  - [ ]* 5.5 Write integration test: forward pass with exactly 32,768 tokens produces valid output
    - Assert at least one non-padding token in output; output terminates with EOS token; no OOM on V100
    - _Requirements: 3.4_

  - [ ]* 5.6 Write integration test: forward+backward memory usage at max sequence length
    - Peak device memory < 95% of 32 GB V100 HBM2 (i.e., < 30.4 GB) with gradient checkpointing enabled
    - _Requirements: 1.4_

- [ ] 6. Vision encoder
  - Implement `VisionEncoder` class: patch extraction (16×16, zero-pad to nearest multiple of 16), learned linear projection to hidden_dim=2,560, 2D learnable positional embeddings, sequence output `Tensor[N_patches, D]`
  - Enforce `max_patches=256`; raise `PatchLimitExceededError` if N_patches > 256 (images producing ≥257 patches are rejected)
  - Implement `build_multimodal_sequence(patch_embeddings, text_embeddings)` that prepends image patches before text tokens
  - _Requirements: 4.1–4.7_

  - [ ] 6.1 Implement `VisionEncoder.encode_image()` with patch extraction, padding, projection, 2D pos encoding
    - Enforce max_patches=256; raise `PatchLimitExceededError` with structured error when exceeded
    - _Requirements: 4.1, 4.2, 4.3, 4.6, 4.7_

  - [ ]* 6.2 Write property test P6: Vision encoder patch tiling
    - **Property 6: Vision Encoder Patch Tiling**
    - **Validates: Requirements 4.1, 4.6**
    - Random image sizes 1–256 patches worth (up to ~256px per side with 16px patches); assert ⌈H/16⌉ × ⌈W/16⌉ non-overlapping patches; full coverage
    - `@settings(max_examples=100)`

  - [ ]* 6.3 Write property test P7: Vision patch embedding dimension consistency
    - **Property 7: Vision Patch Embedding Dimension Consistency**
    - **Validates: Requirements 4.2**
    - Random valid image sizes (within 256-patch limit); assert all patch embeddings have dimension D=2,560
    - `@settings(max_examples=100)`

  - [ ] 6.4 Implement `build_multimodal_sequence()` for image-before-text ordering
    - _Requirements: 4.4_

  - [ ]* 6.5 Write property test P8: Image patch sequence ordering
    - **Property 8: Image Patch Sequence Ordering**
    - **Validates: Requirements 4.4**
    - Random (image, text) pairs within the 256-patch limit; assert no text token at position ≤ last image patch position
    - `@settings(max_examples=100)`

  - [ ]* 6.6 Write unit test: image producing 257 patches raises `PatchLimitExceededError`
    - Construct an image whose dimensions (after padding) yield exactly 257 patches; assert `PatchLimitExceededError` with `{"error": "patch_limit_exceeded", "max_patches": 256, "provided_patches": 257}`
    - _Requirements: 4.7_

- [ ] 7. Checkpoint: model architecture complete
  - Ensure all architecture tests pass, ask the user if questions arise.

- [ ] 8. Data pipeline
  - Implement `DataPipeline` class with methods: `collect_sources()`, `deduplicate()`, `toxicity_filter()`, `quality_filter()`, `process_document()`, `tokenize_and_shard()`
  - MinHash deduplication with Jaccard threshold ≥ 0.80 (use `datasketch` library)
  - Toxicity classifier integration (configurable threshold, default 0.5)
  - Quality heuristics: perplexity check (95th-percentile threshold), length check (50–1,000,000 chars), language-id confidence ≥ 0.90
  - Rejection logging: emit `{doc_id, filter_name, criterion_violated}` for every excluded document
  - Produce byte-identical shard files + manifest file per domain; target 1–2T total tokens across all text modalities
  - _Requirements: 6.1–6.7, 7.1–7.5_

  - [ ] 8.1 Implement MinHash deduplication (`DataPipeline.deduplicate()`)
    - _Requirements: 7.1_

  - [ ]* 8.2 Write property test P11: Deduplication near-duplicate removal
    - **Property 11: Deduplication Near-Duplicate Removal**
    - **Validates: Requirements 7.1**
    - Pairs of documents with generated Jaccard similarity above/below 0.80
    - Assert at most one copy retained when similarity ≥ 0.80; both retained when < 0.80
    - `@settings(max_examples=100)`

  - [ ] 8.3 Implement toxicity filter (`DataPipeline.toxicity_filter()`)
    - _Requirements: 7.2_

  - [ ] 8.4 Implement quality heuristics filter (`DataPipeline.quality_filter()`)
    - Perplexity, length, language-id checks
    - _Requirements: 7.3_

  - [ ]* 8.5 Write property test P12: Quality filter coverage
    - **Property 12: Quality Filter Coverage**
    - **Validates: Requirements 7.3**
    - Documents with each heuristic dimension varied independently; assert failing docs excluded, passing docs retained
    - `@settings(max_examples=100)`

  - [ ] 8.6 Implement rejection logging in `DataPipeline.process_document()`
    - _Requirements: 7.2, 7.4_

  - [ ]* 8.7 Write property test P13: Filter rejection log completeness
    - **Property 13: Filter Rejection Log Completeness**
    - **Validates: Requirements 7.2, 7.4**
    - Random failing documents across all filter types; assert exactly one log entry per rejected doc with all required fields
    - `@settings(max_examples=100)`

  - [ ] 8.8 Implement tokenize-and-shard with manifest output
    - Byte-identical shard files given identical input + config
    - Manifest records token count per domain
    - _Requirements: 7.5_

- [ ] 9. Pre-trainer
  - Implement `PreTrainer` class: NTP cross-entropy loss over non-padding tokens, TP=2/PP=4/DP=5 distributed setup on 40 V100 GPUs (via Megatron-LM or DeepSpeed), AdamW with β1=0.9, β2=0.95, ε=1e-8, weight decay=0.1 (non-embedding/non-bias), gradient clipping (max norm=1.0)
  - LR schedule: linear warmup (0 → 1e-4 over 2,000 steps) + cosine decay (1e-4 → 1e-5)
  - **FP16 mixed precision with dynamic loss scaling**: Use `torch.cuda.amp.GradScaler`; on overflow detected, reduce loss scale by 2× and skip the gradient update without halting training
  - **Gradient checkpointing**: Mandatory; must be enabled throughout all training phases
  - Global batch size fixed at 2,000,000 tokens/step (configurable within [1M, 4M])
  - Checkpoint saving: at **100B tokens**, **300B tokens**, and training end; retry-on-failure (3 retries, exponential backoff)
  - Integrate MoE auxiliary loss into total training loss
  - _Requirements: 8.1–8.8, 9.1–9.7_

  - [ ] 9.1 Implement AdamW optimizer setup and LR schedule (warmup + cosine decay)
    - _Requirements: 9.1–9.6_

  - [ ] 9.2 Implement NTP cross-entropy loss computation with MoE auxiliary loss integration
    - _Requirements: 8.1_

  - [ ] 9.3 Implement distributed training setup (TP=2, PP=4, DP=5 on 40 V100 GPUs)
    - Configure tensor parallelism TP=2 (within each node via NVLink, 2 GPUs/node)
    - Configure pipeline parallelism PP=4 (across 4 nodes via InfiniBand)
    - Configure data parallelism DP=5 (5 replica groups × 8 GPUs = 40 GPUs total)
    - _Requirements: 8.4_

  - [ ] 9.4 Implement checkpoint save logic with retry (3 attempts, exponential backoff) and training halt on failure
    - Milestone triggers: 100B tokens and 300B tokens (not 1T or 5T)
    - _Requirements: 8.5, 8.6, 8.7, 8.8_

  - [ ]* 9.5 Write unit tests: checkpoint milestones (100B tokens, 300B tokens, end-of-training)
    - Simulate token counters crossing 100_000_000_000 and 300_000_000_000; assert checkpoints saved before next step
    - _Requirements: 8.5, 8.6, 8.7_

  - [ ]* 9.6 Write unit test: checkpoint save failure → halt after 3 retries
    - Mock checkpoint save to fail 3 times; assert training halts and error emitted
    - _Requirements: 8.8_

  - [ ] 9.7 Implement FP16 dynamic loss scaling (GradScaler integration)
    - Wrap optimizer step with `GradScaler.scale()`, `GradScaler.step()`, `GradScaler.update()`
    - On overflow: `GradScaler` automatically reduces loss scale by 2× and skips gradient update
    - Log each overflow event and the updated loss scale value
    - _Requirements: 9.7_

  - [ ]* 9.8 Write unit test: FP16 overflow handling (loss scale reduction + gradient skip)
    - Simulate a gradient overflow condition; assert: (a) gradient update is skipped, (b) loss scale is halved, (c) training does not halt
    - _Requirements: 9.7_

- [ ] 10. Checkpoint: training infrastructure complete
  - Ensure all training infrastructure tests pass, ask the user if questions arise.

- [ ] 11. Fine-tuner (Constitutional AI + RLHF + Agentic reasoning)
  - Implement `FineTuner` class with two sequential stages:
    - Stage 1: Constitutional AI / SFT on 800M–1.2B curated tokens, peak LR ≤ 1e-5
    - Stage 2: RLHF with `RewardModel` (BERT-like) trained on ≥100K preference pairs, PPO or DPO fine-tuning for ≤1% of pre-training gradient steps
  - Implement early stopping: halt and restore best checkpoint if avg reward on held-out 5K set drops below SFT baseline
  - Agentic reasoning fine-tuning data: ≥10% of fine-tuning token budget covers task delegation, planning, self-verification
  - Implement inference-time planning mode: decompose tasks into ordered subtask sequences; emit intermediate steps; self-revision loop (≤3 attempts per inconsistency)
  - FP16 + gradient checkpointing must remain enabled during fine-tuning phases (same V100 constraints apply)
  - _Requirements: 10.1–10.5, 11.1–11.5_

  - [ ] 11.1 Implement Constitutional AI / SFT training stage
    - _Requirements: 10.1, 10.3_

  - [ ] 11.2 Implement `RewardModel` and RLHF fine-tuning (PPO/DPO) with early stopping
    - _Requirements: 10.2, 10.3, 10.4, 10.5_

  - [ ]* 11.3 Write unit test: reward drop below SFT baseline triggers halt and checkpoint restore
    - Simulate reward dropping below baseline; assert halt and best checkpoint restored
    - _Requirements: 10.5_

  - [ ] 11.4 Implement agentic planning mode: task decomposition, intermediate step emission, self-revision loop
    - _Requirements: 11.1–11.5_

  - [ ]* 11.5 Write property test P15: Agentic output intermediate steps
    - **Property 15: Agentic Output Intermediate Steps**
    - **Validates: Requirements 11.2, 11.3**
    - Multi-step task prompts with varying step complexity; assert ≥2 intermediate reasoning steps before final answer
    - `@settings(max_examples=100)`

  - [ ]* 11.6 Write unit test: self-revision loop ≤3 attempts, emits best output with inconsistency indicator
    - Input causing persistent contradiction; assert ≤3 revision attempts and indicator in output
    - _Requirements: 11.4, 11.5_

- [ ] 12. Safety classifier
  - Implement `SafetyClassifier` (BERT-base, ≤110M params): binary classification `flagged` / `not-flagged`
  - Implement `classify(prompt: str) -> SafetyClassifierResult` with latency **≤200ms** on V100 inference hardware (relaxed from ≤50ms due to V100 vs. H100 inference speed)
  - Implement training and retraining scripts (no LLM weight changes during retraining)
  - Precision ≥ 0.90, Recall ≥ 0.85 enforcement: retraining script must re-evaluate on held-out set and reject deployment if thresholds not met
  - _Requirements: 12.1–12.5_

  - [ ] 12.1 Implement `SafetyClassifier` model and `classify()` method
    - _Requirements: 12.1, 12.2_

  - [ ] 12.2 Implement retraining script with threshold enforcement (no LLM weight changes)
    - _Requirements: 12.4_

  - [ ]* 12.3 Write integration test: Safety Classifier latency p99 ≤ 200ms on V100
    - Benchmark 1,000 classify() calls on V100 hardware; assert p99 latency ≤ 200ms
    - _Requirements: 12.2_

  - [ ]* 12.4 Write integration test: precision ≥ 0.90, recall ≥ 0.85 on held-out 2,000-example set
    - _Requirements: 12.3_

  - [ ]* 12.5 Write integration test: retrain with 500 new examples; verify LLM weights unchanged; re-evaluate P/R
    - _Requirements: 12.4_

- [ ] 13. Gating router
  - Implement `GatingRouter.route(request: APIRequest) -> APIResponse` as described in the sequence diagram
  - Invoke `SafetyClassifier` before forwarding; enforce **400ms timeout** (not 100ms — accounts for V100 inference latency); treat timeout/error as flagged
  - Route to unconstrained LLM if `not-flagged`; route to constrained model if `flagged`, timeout, or error
  - Log `classifier_result` and `routed_to` on every response
  - _Requirements: 12.5, 13.1–13.5_

  - [ ] 13.1 Implement `GatingRouter` with classify-then-route logic and 400ms timeout/error fallback
    - _Requirements: 13.1, 13.2, 13.3, 13.5_

  - [ ]* 13.2 Write property test P14: Gating router correct routing
    - **Property 14: Gating Router Correct Routing**
    - **Validates: Requirements 12.5, 13.1, 13.2, 13.3, 13.5**
    - Mocked classifier responses (flagged / not-flagged / timeout at >400ms / error); assert routing invariants hold in all cases
    - `@settings(max_examples=100)`

  - [ ]* 13.3 Write integration test: unconstrained routing rate ≥ 95% over simulated 30-day window
    - _Requirements: 13.4_

- [ ] 14. API server and monitoring
  - Implement FastAPI (or equivalent) server with endpoints `/v1/completions`, `/v1/chat`, `/v1/agentic`
  - Text + optional image inputs; text output
  - Request/response logging: payload, timestamp, endpoint ID, client ID; 30-day retention
  - Fixed temperature per API version config (no runtime adjustment)
  - Planning mode enabled by default on `/v1/agentic`
  - Implement flagging-rate monitor: compare rolling 7-day flagging rate to first-7-day baseline; alert ops team within 1 hour when ≥50% increase sustained ≥1 hour over any rolling 7-day window
  - _Requirements: 15.1–15.6_

  - [ ] 14.1 Implement API server with versioned endpoints, input validation, and request/response logging
    - _Requirements: 15.1, 15.2, 15.3, 15.4_

  - [ ] 14.2 Implement flagging-rate monitor and alert mechanism
    - _Requirements: 15.5_

  - [ ]* 14.3 Write unit test: flagging rate spike triggers alert within 1 hour
    - Simulate flagging rate spike; assert alert emitted within simulated 1-hour window
    - _Requirements: 15.5_

  - [ ]* 14.4 Write integration test: continuous retraining does not drop API SLA; deployment within 24 hours
    - _Requirements: 15.6_

- [ ] 15. Integration: wire all components end-to-end
  - Wire `Tokenizer` → `VisionEncoder` → `TransformerDecoder` (with 64-expert MoE layers + 32K hybrid attention) → `GatingRouter` → `SafetyClassifier` → `API Server`
  - Wire `DataPipeline` → `PreTrainer` (FP16 + gradient checkpointing, TP=2/PP=4/DP=5, 40 V100 GPUs) → `FineTuner` → aligned model artifacts
  - Verify `APIRequest` → `APIResponse` full round-trip with text-only and multimodal inputs (within 256-patch image limit)
  - _Requirements: all_

  - [ ] 15.1 Wire inference pipeline: Tokenizer → VisionEncoder → TransformerDecoder → GatingRouter → API Server
    - _Requirements: 1.1–1.10, 3.1–3.6, 4.1–4.7, 5.1–5.5, 12.5, 13.1–13.5, 15.1–15.4_

  - [ ] 15.2 Wire training pipeline: DataPipeline → PreTrainer → FineTuner → model artifact output
    - _Requirements: 6.1–6.7, 7.1–7.5, 8.1–8.8, 9.1–9.7, 10.1–10.5, 11.1–11.5_

  - [ ]* 15.3 Write integration test: text-only and multimodal end-to-end API round-trip
    - _Requirements: 4.4, 4.5, 15.1_

- [ ] 16. Final checkpoint — ensure all tests pass
  - Ensure all unit, property, integration, and smoke tests pass. Ask the user if questions arise.

---

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- All 15 correctness properties from the design document have corresponding property-based test sub-tasks using `hypothesis`
- Each property test requires `@settings(max_examples=100)` and a comment tag: `# Feature: llm-10b-build, Property N: <property_title>`
- Checkpoints (tasks 7, 10, 16) are integration gates — all preceding tests should pass before continuing
- Property tests are placed close to their corresponding implementation tasks to catch errors early
- **V100 hardware constraints that affect every task**:
  - **No FlashAttention-2**: FA-2 requires CUDA CC 8.0+ (Ampere); V100 is CC 7.0. Use xformers memory-efficient attention or FlashAttention-1 (v1 API) throughout.
  - **No BF16**: V100 CC 7.0 does not support BF16 natively. All mixed-precision training uses FP16 with `torch.cuda.amp.GradScaler` (dynamic loss scaling). On overflow, scale is halved and the gradient update is skipped.
  - **Gradient checkpointing is mandatory**: Without it, 32K-length activations exceed 32 GB V100 HBM2. `use_gradient_checkpointing=True` is enforced in `ModelConfig.__post_init__` and must remain enabled through pre-training and fine-tuning.
- Integration tests (tasks 5.3, 5.4, 5.5, 5.6, 12.3–12.5, 13.3, 14.4) require production-grade V100 hardware or a sufficiently capable simulation environment
- Pre-training at full scale (40 V100 GPUs, 10²¹–10²² FLOPs, 2–4 weeks) is a cluster operation; the code tasks here implement the full software stack, but actual training runs require cluster provisioning outside the scope of a coding agent
- Red-teaming (Requirement 14) is a human process and is intentionally excluded per the "coding tasks only" constraint
- Checkpoint milestones are **100B and 300B tokens** (scaled from the original 1T/5T to match the 1–2T total corpus of the Param Ganga spec)
- Safety classifier latency budget is **≤200ms** on V100 (not ≤50ms); gating router timeout is **400ms** (not 100ms)

---

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "2.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "3.1"] },
    { "id": 3, "tasks": ["3.2", "3.4"] },
    { "id": 4, "tasks": ["3.3", "3.5", "8.1"] },
    { "id": 5, "tasks": ["3.6", "3.7", "3.8", "4.1", "8.2", "8.3", "8.4"] },
    { "id": 6, "tasks": ["4.2", "4.3", "4.4", "8.5", "8.6"] },
    { "id": 7, "tasks": ["4.5", "5.1", "6.1", "8.7", "8.8"] },
    { "id": 8, "tasks": ["5.2", "6.2", "6.3", "6.4", "9.1", "9.2"] },
    { "id": 9, "tasks": ["5.3", "5.4", "5.5", "5.6", "6.5", "6.6", "9.3"] },
    { "id": 10, "tasks": ["9.4", "9.7"] },
    { "id": 11, "tasks": ["9.5", "9.6", "9.8", "11.1"] },
    { "id": 12, "tasks": ["11.2"] },
    { "id": 13, "tasks": ["11.3", "11.4"] },
    { "id": 14, "tasks": ["11.5", "11.6", "12.1"] },
    { "id": 15, "tasks": ["12.2"] },
    { "id": 16, "tasks": ["12.3", "12.4", "12.5", "13.1"] },
    { "id": 17, "tasks": ["13.2", "13.3"] },
    { "id": 18, "tasks": ["14.1"] },
    { "id": 19, "tasks": ["14.2"] },
    { "id": 20, "tasks": ["14.3", "14.4", "15.1", "15.2"] },
    { "id": 21, "tasks": ["15.3"] }
  ]
}
```
