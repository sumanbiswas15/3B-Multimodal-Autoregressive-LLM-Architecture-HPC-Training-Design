# Requirements Document

## Introduction

This document specifies the requirements for building a ~3-billion-parameter, multimodal, autoregressive large language model (LLM) supporting a 32,768-token context window, targeting the **Param Ganga HPC cluster at IIT Roorkee**. The cluster comprises 20 GPU-only compute nodes, each equipped with 2 × Intel Xeon Skylake 6248 (20 cores, 2.5 GHz), 192 GB RAM, and 2 × NVIDIA V100 SXM2 32 GB GPUs — totalling 40 V100 GPUs and approximately 5,000 TFLOPS FP16 cluster-wide.

The model is developed across five phases: architecture design, data collection and preprocessing, base pre-training, alignment fine-tuning, and safety and deployment. All design decisions are adapted to the V100 hardware constraints: 32 GB HBM2 per card, FP16-only mixed precision (BF16 is unavailable on CUDA Compute Capability 7.0), no FlashAttention-2 support (FlashAttention-1 / xformers memory-efficient attention is used instead), mandatory gradient checkpointing, and a 40-GPU total cluster limit.

The model targets strong language understanding, generation, code reasoning, multimodal (vision-language) tasks, and agentic behaviors, with robust safety properties at inference time.

---

## Glossary

- **LLM**: Large Language Model — the ~3B-parameter autoregressive Transformer being built.
- **Transformer**: The neural network backbone based on multi-head self-attention.
- **MoE**: Mixture-of-Experts — a sparsely-activated feed-forward layer where only a subset of experts processes each token.
- **RoPE**: Rotary Position Embedding — a positional encoding scheme applied to attention queries and keys.
- **ALiBi**: Attention with Linear Biases — an alternative positional encoding that biases attention scores by distance.
- **SwiGLU**: A gated linear unit activation function combining Swish and GLU, commonly used in Transformer FFN layers.
- **GELU**: Gaussian Error Linear Unit activation function.
- **Chunked_Attention**: An attention mechanism that processes long sequences in fixed-size chunks.
- **Sliding_Window_Attention**: An attention mechanism where each token attends only to a local window of surrounding tokens.
- **Hierarchical_Attention**: An attention mechanism that compresses distant context into summary representations for long-range attention.
- **Vision_Encoder**: The component that converts image pixels into patch embeddings for multimodal processing.
- **Tokenizer**: The shared subword tokenizer that converts raw text (and multimodal signals) into discrete token IDs.
- **Pre_Trainer**: The training subsystem responsible for base pre-training via next-token prediction.
- **Fine_Tuner**: The training subsystem responsible for alignment fine-tuning via supervised and reinforcement learning.
- **Safety_Classifier**: A small BERT-like classifier that identifies risky or harmful content in prompts and outputs.
- **Gating_Router**: The inference-time component that routes requests to the constrained or unconstrained model based on Safety_Classifier output.
- **Data_Pipeline**: The preprocessing and deduplication system that produces cleaned, tokenized training corpora.
- **Checkpoint**: A saved snapshot of model weights and optimizer state at a specific training step.
- **RLHF**: Reinforcement Learning from Human Feedback — a fine-tuning method using reward models trained on human preference labels.
- **Constitutional_AI**: A fine-tuning technique that uses a set of principles to guide model behavior via self-critique and revision.
- **AdamW**: The Adam optimizer variant with decoupled weight decay.
- **FLOPs**: Floating-point operations — used to measure total training compute.
- **Red_Team**: A group conducting adversarial testing to identify safety and robustness failures in the LLM.
- **Common_Crawl**: A publicly available web-scale text corpus.
- **LAION**: A large open dataset of image-text pairs used for multimodal training.
- **API_Server**: The production serving layer that exposes the LLM's capabilities via a network API.
- **V100**: NVIDIA Tesla V100 SXM2 GPU with 32 GB HBM2, 125 TFLOPS FP16, CUDA Compute Capability 7.0.
- **Param_Ganga**: The IIT Roorkee HPC cluster hosting 20 nodes × 2 V100 GPUs = 40 V100 GPUs total.

---

## Requirements

---

### Requirement 1: Transformer Architecture

**User Story:** As an ML engineer, I want a well-specified autoregressive Transformer architecture, so that the model has a proven and scalable backbone for language modeling on the Param Ganga V100 cluster.

#### Acceptance Criteria

1. THE LLM SHALL use an autoregressive Transformer decoder backbone with between 28 and 48 layers.
2. THE LLM SHALL use between 16 and 32 multi-head attention heads per layer.
3. THE LLM SHALL use either GELU or SwiGLU as its feed-forward network activation function.
4. WHEN the forward and backward passes are executed on the target training hardware, THE LLM SHALL complete both passes without exceeding 95% of the available device memory on any single V100 SXM2 GPU (32 GB HBM2), using xformers memory-efficient attention or FlashAttention-1 (not FlashAttention-2, which is unsupported on CUDA Compute Capability 7.0) rather than materializing the full O(n²) attention matrix, with gradient checkpointing enabled.
5. THE LLM SHALL apply either RoPE or ALiBi positional encodings to all attention layers.
6. THE LLM SHALL use no weight tying between the input embedding matrix and the output projection matrix.
7. WHEN a token sequence is processed, THE LLM SHALL produce, at each position, a probability distribution over the vocabulary where each value is in [0.0, 1.0] and all values sum to 1.0 (± 1×10⁻⁵ floating-point tolerance).
8. THE LLM SHALL use a vocabulary size of between 32,000 and 256,000 tokens.
9. THE LLM SHALL use a hidden dimension of at least 2,048 and support input sequences of up to 32,768 tokens.
10. THE LLM SHALL use a feed-forward expansion ratio of at least 4× the hidden dimension in each dense feed-forward layer.

---

### Requirement 2: Mixture-of-Experts Feed-Forward Layers

**User Story:** As an ML engineer, I want a Mixture-of-Experts feed-forward layer, so that the model achieves high parameter count while keeping per-token compute tractable within the V100 32 GB memory budget.

#### Acceptance Criteria

1. THE LLM SHALL include MoE layers with a total of at least 64 experts per MoE layer.
2. WHEN a token is processed by an MoE layer, THE LLM SHALL activate exactly k experts per token, where k is a positive integer satisfying 1 ≤ k ≤ total experts per MoE layer, and k is fixed at architecture configuration time.
3. THE LLM SHALL use a learned router that produces a scalar score per expert for each token, selects the top-k experts by score, and combines their outputs as a weighted sum using the normalized router scores.
4. THE LLM SHALL apply no dropout within MoE layers during training.
5. IF any expert's fraction of assigned tokens during a training step deviates from the uniform fraction (1/N, where N is the total number of experts) by more than a configurable tolerance threshold, THEN THE Pre_Trainer SHALL apply an auxiliary load-balancing loss whose weight is independently configurable, so that each expert's token fraction remains within the configured tolerance of 1/N.
6. Load imbalance SHALL be defined as the condition where any single expert's fraction of tokens assigned in a training step deviates from 1/N by more than the configured tolerance threshold.

---

### Requirement 3: Long-Context Attention (32K Tokens)

**User Story:** As a product engineer, I want the model to support a 32,768-token context window, so that the LLM can process long documents, extended conversations, and moderate-size codebases in a single pass within the V100 memory budget.

#### Acceptance Criteria

1. THE LLM SHALL support input sequences of up to 32,768 tokens without truncation.
2. WHEN queried about a fact or token present in the first 10% of a 32,768-token input, THE LLM SHALL produce the correct fact or token in its output with accuracy no lower than its accuracy on an equivalent query over a 4,096-token input containing the same fact, as measured on a held-out long-context retrieval benchmark.
3. WHEN asked to reason over information present in both the first 10% and the last 10% of a 32,768-token input, THE LLM SHALL produce an output consistent with both information sources, as evaluated by a reference answer on a held-out cross-context coherence benchmark.
4. WHEN a sequence of exactly 32,768 tokens is provided, THE LLM SHALL produce a valid output — defined as an output containing at least one non-padding token and terminating with the end-of-sequence token — without raising an out-of-memory error on the target hardware configuration specified in the deployment documentation.
5. WHEN an input sequence exceeds 32,768 tokens, THE LLM SHALL reject the request with a well-defined error response indicating the maximum supported sequence length, without silently truncating the input.
6. THE LLM SHALL maintain causal masking across all attention mechanisms, such that for any token at position i, its attention weight to any token at position j > i is exactly 0.0, verifiable by inspecting the attention weight matrix on a test input.

---

### Requirement 4: Multimodal Vision Integration

**User Story:** As a product engineer, I want the model to process images alongside text, so that the LLM can answer questions about visual content and handle vision-language tasks within the V100 memory constraints.

#### Acceptance Criteria

1. THE Vision_Encoder SHALL partition each input image into non-overlapping 16×16 pixel patches.
2. THE Vision_Encoder SHALL project each image patch into an embedding vector of the same dimensionality as the text token embeddings.
3. THE Vision_Encoder SHALL assign a 2D positional encoding to each patch embedding reflecting its row and column position in the source image.
4. WHEN an input contains both image patches and text tokens, THE LLM SHALL place all image patch embeddings before the text token embeddings in the input sequence and process them through the shared Transformer layers with full global self-attention.
5. WHEN a multimodal input containing at least one image and at least one text token is provided, THE LLM SHALL produce an output that contains at least one non-padding token and terminates with the end-of-sequence token, within the same latency bounds as a text-only input of equivalent total token count.
6. WHEN an input image's height or width is not an exact multiple of 16 pixels, THE Vision_Encoder SHALL pad the image to the nearest multiple of 16 in each dimension before patch extraction, and SHALL NOT raise an error or produce undefined behavior.
7. WHEN an input image would produce more than 256 patches after padding, THE Vision_Encoder SHALL reject the image with a well-defined error response indicating the maximum supported patch count, without silently dropping patches.

---

### Requirement 5: Shared Subword Tokenizer

**User Story:** As an ML engineer, I want a shared subword tokenizer for all modalities, so that text, code, and multimodal signals use a unified vocabulary.

#### Acceptance Criteria

1. THE Tokenizer SHALL use a subword algorithm (BPE or Unigram) trained on a representative sample of the full pre-training corpus containing at least 10 billion tokens, with no single modality (text, code, or multilingual) comprising more than 70% of the training sample.
2. THE Tokenizer SHALL encode any valid UTF-8 string into a sequence of token IDs without data loss.
3. THE Tokenizer SHALL satisfy the round-trip property: for any valid UTF-8 string s that does not contain special control tokens, decode(encode(s)) == s.
4. THE Tokenizer SHALL support a vocabulary size of at least 32,000 tokens and at most 256,000 tokens.
5. WHEN a string contains a byte sequence not representable by a vocabulary token, THE Tokenizer SHALL fall back to byte-level encoding using 256 reserved byte tokens (one per byte value 0x00–0xFF), such that every valid byte sequence is encodable without raising an error.

---

### Requirement 6: Data Collection Volumes

**User Story:** As an ML researcher, I want a clearly specified data collection target, so that the pre-training corpus is large and diverse enough to support strong generalization for a ~3B parameter model by Chinchilla scaling.

#### Acceptance Criteria

1. THE Data_Pipeline SHALL collect between 600 billion and 1.2 trillion tokens of web text from sources including Common_Crawl, news, and forum content.
2. THE Data_Pipeline SHALL collect between 100 billion and 400 billion tokens of academic, book, and fiction content.
3. THE Data_Pipeline SHALL collect between 50 billion and 200 billion tokens of code sourced from public repositories such as GitHub.
4. THE Data_Pipeline SHALL collect between 5 billion and 50 billion tokens of specialized corpora covering genomics, chemistry, and biomedical domains, with each domain contributing at least 1 billion tokens.
5. THE Data_Pipeline SHALL collect between 10 million and 100 million images paired with between 10 billion and 100 billion caption tokens in a LAION-style format.
6. THE Data_Pipeline SHALL collect between 800 million and 1.2 billion tokens of alignment and question-answering data for use in fine-tuning.
7. THE Data_Pipeline SHALL target a combined total of between 1 trillion and 2 trillion tokens across all text modalities (web, academic, code, specialized, and alignment) before deduplication; image caption tokens are excluded from this count.

---

### Requirement 7: Data Preprocessing and Quality Control

**User Story:** As an ML researcher, I want deduplicated, filtered, and quality-controlled training data, so that the model does not memorize duplicate content or learn from toxic or low-quality text.

#### Acceptance Criteria

1. THE Data_Pipeline SHALL deduplicate the corpus at the document level using an exact-match or MinHash near-duplicate detection algorithm with a Jaccard similarity threshold of at least 0.80; any two documents with Jaccard similarity ≥ 0.80 SHALL result in all but one copy being removed before tokenization.
2. THE Data_Pipeline SHALL apply toxicity filtering using a classifier that produces a toxicity score in [0.0, 1.0], and SHALL remove any document whose score meets or exceeds a configurable threshold (default 0.5) to exclude documents classified as harmful, hateful, or explicit.
3. THE Data_Pipeline SHALL apply the following quality-control heuristics and remove any document that fails any one of them: (a) perplexity under a reference language model exceeds the 95th-percentile perplexity of a held-out clean corpus; (b) document length after whitespace normalization is fewer than 50 characters or more than 1,000,000 characters; (c) language identification confidence score is below 0.90 for the document's declared language.
4. WHEN a document fails any quality or toxicity filter, THE Data_Pipeline SHALL exclude the document from the training corpus and emit a log entry containing the document identifier, the name of the filter that excluded it, and the specific criterion violated.
5. THE Data_Pipeline SHALL produce a final tokenized corpus as byte-identical shard files given identical inputs and configuration, and SHALL produce a manifest file recording the number of tokens per domain.

---

### Requirement 8: Base Pre-Training Objective and Compute

**User Story:** As an ML researcher, I want a well-defined pre-training objective and compute budget, so that the model achieves strong language modeling performance within the Param Ganga cluster's 40 V100 GPU infrastructure.

#### Acceptance Criteria

1. THE Pre_Trainer SHALL train the LLM using cross-entropy next-token prediction loss computed over all non-padding tokens.
2. THE Pre_Trainer SHALL use a global batch size of between 1 million and 4 million tokens per step, with the exact value fixed in the training configuration file before training begins and held constant for the entire run.
3. THE Pre_Trainer SHALL target a total training compute of between 10^21 and 10^22 floating-point operations (FLOPs), achievable in 2 to 4 weeks at approximately 5,000 TFLOPS FP16 cluster-wide with 30–40% hardware utilization.
4. THE Pre_Trainer SHALL distribute training across the 40 V100 SXM2 GPUs of the Param Ganga cluster using tensor parallelism (TP=2, within each node via NVLink), pipeline parallelism (PP=4, across 4 nodes via InfiniBand), and data parallelism (DP=5, across 5 replica groups), yielding 20 nodes × 2 GPUs = 40 GPUs total.
5. WHEN cumulative training tokens processed first exceeds 100 billion, THE Pre_Trainer SHALL save a Checkpoint containing full model weights, optimizer state, and the current token count before the next training step begins.
6. WHEN cumulative training tokens processed first exceeds 300 billion, THE Pre_Trainer SHALL save a Checkpoint containing full model weights, optimizer state, and the current token count before the next training step begins.
7. THE Pre_Trainer SHALL save a final Checkpoint containing full model weights, optimizer state, and total token count upon normal training termination, regardless of the token count reached.
8. WHEN a Checkpoint save fails, THE Pre_Trainer SHALL retry the save up to 3 times before halting training and emitting an error with the failure reason, to prevent silent data loss.

---

### Requirement 9: Optimizer and Learning Rate Schedule

**User Story:** As an ML researcher, I want a specified optimizer configuration and learning rate schedule, so that training is stable and converges to a high-quality solution under FP16 mixed precision with loss scaling on V100 GPUs.

#### Acceptance Criteria

1. THE Pre_Trainer SHALL use the AdamW optimizer with β1 = 0.9, β2 = 0.95, and ε = 1×10⁻⁸.
2. THE Pre_Trainer SHALL apply a weight decay of 0.1 to all non-embedding, non-bias parameters.
3. THE Pre_Trainer SHALL use a peak learning rate of 1×10⁻⁴ during pre-training.
4. THE Pre_Trainer SHALL apply a cosine learning rate decay schedule that begins after the warmup phase, decaying from the peak learning rate of 1×10⁻⁴ to a final learning rate of 1×10⁻⁵ by the end of training.
5. THE Pre_Trainer SHALL apply gradient clipping with a maximum gradient norm of 1.0 to prevent training instability.
6. THE Pre_Trainer SHALL apply a linear learning rate warmup for the first 2,000 training steps, during which the learning rate rises linearly from 0 to the peak learning rate of 1×10⁻⁴.
7. THE Pre_Trainer SHALL use FP16 mixed precision training with dynamic loss scaling, given that BF16 is not natively supported on CUDA Compute Capability 7.0 (V100). WHEN a loss scale overflow is detected, THE Pre_Trainer SHALL reduce the loss scale by a factor of 2 and skip the current gradient update without halting training.

---

### Requirement 10: Alignment Fine-Tuning via Constitutional AI and RLHF

**User Story:** As a product engineer, I want the model to be aligned to human preferences and follow instructions helpfully, so that the deployed LLM behaves safely and usefully in interactive settings.

#### Acceptance Criteria

1. THE Fine_Tuner SHALL apply Constitutional AI fine-tuning on between 800 million and 1.2 billion curated question-answering and chat tokens drawn from the alignment dataset collected in Requirement 6.
2. THE Fine_Tuner SHALL apply RLHF using a reward model trained on at least 100,000 human preference-labeled response pairs, where each pair consists of two responses to the same prompt and a human label indicating which response is preferred.
3. THE Fine_Tuner SHALL use a peak learning rate of no more than 1×10⁻⁵ during RLHF, which is at least one order of magnitude lower than the pre-training peak learning rate.
4. THE Fine_Tuner SHALL train for no more than 1% of the total gradient steps used during base pre-training.
5. WHEN the fine-tuned model's average reward on a held-out preference evaluation set of at least 5,000 examples drops below the average reward of the supervised fine-tuning baseline on the same set, THE Fine_Tuner SHALL halt training and restore the Checkpoint with the highest average reward observed during the current fine-tuning run.

---

### Requirement 11: Agentic Reasoning Capabilities

**User Story:** As a product engineer, I want the model to reason agentically with planning and self-verification, so that the LLM can complete multi-step tasks without explicit user prompting at every step.

#### Acceptance Criteria

1. THE Fine_Tuner SHALL include agentic reasoning fine-tuning data covering task delegation, multi-step planning, and self-verification behaviors, comprising at least 10% of the total fine-tuning token budget.
2. WHEN a multi-step task requiring at least two distinct reasoning steps is presented to the LLM at inference time, THE LLM SHALL produce at least 2 distinct intermediate reasoning steps before emitting a final answer, without requiring explicit chain-of-thought prompting from the user.
3. WHILE planning mode is active at inference, THE LLM SHALL decompose the task into an ordered sequence of subtasks and emit that sequence before beginning execution of any subtask.
4. IF THE LLM detects a contradiction or factual inconsistency in one of its own intermediate outputs during a multi-step task, THEN THE LLM SHALL revise that intermediate output before proceeding to the next step.
5. THE LLM SHALL attempt self-revision of the same intermediate output no more than 3 consecutive times; IF the contradiction or inconsistency is not resolved after 3 revision attempts, THEN THE LLM SHALL emit the best available intermediate output with an indication that the inconsistency was not resolved, and proceed to the next step.

---

### Requirement 12: Safety Classifier Design

**User Story:** As a safety engineer, I want a lightweight safety classifier trained on labeled risky prompts, so that harmful requests can be identified at inference time with low latency overhead on V100 hardware.

#### Acceptance Criteria

1. THE Safety_Classifier SHALL be a BERT-architecture encoder model with no more than 110 million parameters, trained on a labeled dataset of at least 10,000 examples with at least 30% positive (risky/harmful) examples, covering domains including cybersecurity, biology, and general harmful instructions.
2. THE Safety_Classifier SHALL classify any input prompt as either flagged or not-flagged within 200 milliseconds, measured on the same CPU or GPU hardware configuration used for LLM inference in production on the Param Ganga V100 nodes.
3. THE Safety_Classifier SHALL achieve a precision of at least 0.90 and a recall of at least 0.85 on a held-out labeled safety evaluation set containing at least 2,000 examples with at least 30% positive examples.
4. WHEN a new batch of at least 500 labeled safety examples is available, THE Safety_Classifier SHALL support retraining without requiring changes to the LLM weights, and the retrained classifier SHALL meet or exceed the precision and recall thresholds specified in criterion 3 on the same held-out evaluation set.
5. WHEN THE Safety_Classifier returns a flagged result for an input prompt, THE Gating_Router SHALL route that request to the constrained model and SHALL NOT forward the request to the unconstrained LLM.

---

### Requirement 13: Inference-Time Gating and Routing

**User Story:** As a safety engineer, I want an inference-time gating mechanism, so that flagged requests are routed to a constrained model while the vast majority of safe sessions proceed without restriction.

#### Acceptance Criteria

1. THE Gating_Router SHALL invoke THE Safety_Classifier on every incoming API request before forwarding the request to either the unconstrained LLM or the constrained model.
2. WHEN THE Safety_Classifier returns a not-flagged result, THE Gating_Router SHALL forward the request to the unconstrained LLM and return its response to the caller.
3. WHEN THE Safety_Classifier returns a flagged result, THE Gating_Router SHALL route the request to the constrained model configured with refusal logic, and SHALL return the constrained model's response — which may include a refusal — to the caller.
4. THE Gating_Router SHALL ensure that at least 95% of sessions in production are served by the unconstrained LLM, measured over any rolling 30-day window.
5. WHEN THE Safety_Classifier fails to return a result within 400 milliseconds or returns an error, THE Gating_Router SHALL treat the request as flagged and route it to the constrained model, so that classifier failure never results in an unsafe request reaching the unconstrained LLM.

---

### Requirement 14: Red-Teaming and Adversarial Evaluation

**User Story:** As a safety engineer, I want systematic red-teaming before deployment, so that the model's failure modes are identified and mitigated before public release.

#### Acceptance Criteria

1. THE Red_Team SHALL conduct at least 1,000 hours of structured adversarial prompt testing — defined as testing executed against a documented test plan specifying pre-defined attack categories and explicit pass/fail exit criteria — on the fine-tuned model, using at least 500 unique adversarial prompts, before production deployment is approved.
2. THE Red_Team SHALL include at least 10 crowd-sourced contributors generating adversarial prompts across at least the following categories: jailbreaks, harmful instructions, misinformation generation, and privacy violations.
3. WHEN THE Red_Team identifies a new adversarial prompt pattern that bypasses THE Safety_Classifier, THE Safety_Classifier training data SHALL be updated with labeled examples of that pattern within 5 business days of identification.
4. THE Red_Team SHALL produce a written report before production deployment is approved, documenting identified vulnerabilities classified by severity (Critical, High, Medium, Low), mitigations applied for each vulnerability, and a statement of residual risks for all unmitigated vulnerabilities.

---

### Requirement 15: API Deployment and Logging

**User Story:** As a platform engineer, I want a production API with mandatory data logging and continuous monitoring, so that post-release behavior can be audited and safety classifiers can be retrained on real-world distribution.

#### Acceptance Criteria

1. THE API_Server SHALL expose the LLM's capabilities via a versioned network API supporting at least text and image inputs and text outputs.
2. THE API_Server SHALL retain logs of all requests and responses — including request payload, response payload, timestamp, endpoint identifier, and client identifier — for a mandatory minimum period of 30 days.
3. THE API_Server SHALL NOT apply dynamic temperature adjustments at runtime; the generation temperature SHALL be fixed per API configuration and SHALL remain constant for the lifetime of a deployed API version.
4. THE API_Server SHALL operate with planning mode enabled by default for all agentic API endpoints, defined as endpoints that accept multi-step tool-use or autonomous task instructions.
5. WHEN post-release monitoring detects that the Safety_Classifier flagging rate over any 7-day rolling window has increased by 50% or more relative to the baseline flagging rate established in the first 7 days post-launch, and the elevated rate has been sustained for at least 1 hour, THE API_Server SHALL alert the operations team within 1 hour of that condition being met.
6. THE Safety_Classifier SHALL support continuous retraining from logged production data such that the API_Server's availability does not drop below its normal SLA during retraining, and retraining SHALL complete and the updated classifier SHALL be deployed within 24 hours of new training data becoming available.
