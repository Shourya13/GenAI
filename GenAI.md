# GenAI Core Concepts

This document summarizes foundational concepts in modern Generative AI (GenAI) systems, focusing on how language models process and manage information. The goal of these notes is interview preparation, so the structure is kept compact, content-heavy, and easy to revise quickly.

## Large Language Models (LLMs)

A **Large Language Model (LLM)** is a deep learning model trained on massive amounts of text data so it can understand, generate, summarize, translate, and reason over human language. "Large" refers both to the scale of training data and to the number of learned parameters. Before LLMs, most NLP systems required separate task-specific models. LLMs changed this by making one pretrained model flexible enough to handle many language tasks through prompting and post-training alignment.

At the center of most modern GenAI systems are **LLMs**. They are used because they can understand prompts, reason over text patterns, generate natural language, summarize information, answer questions, write code, and interact conversationally.

### LLMs: Why, Where, How, When

- **Why** -> to automate or assist tasks involving language, reasoning, summarization, generation, and Q&A
- **Where** -> chat systems, search copilots, enterprise assistants, coding tools, research tools, workflow automation
- **How** -> by sending prompts, instructions, context, and sometimes retrieved knowledge to the model
- **When** -> when rule-based systems are too rigid and the task needs flexible language understanding or generation

### Architecture

All modern LLMs are built on the **Transformer architecture**. Before transformers, **RNNs** and **LSTMs** processed tokens sequentially and struggled with long-range dependencies. Transformers process tokens in parallel and use **attention** to directly model relationships between tokens, even when they are far apart.

Core components of an LLM:

- **Tokenizer** -> converts raw text into token IDs
- **Embedding Layer** -> maps each token ID to a dense vector
- **Positional information** -> added because transformers process tokens in parallel
- **Transformer Blocks x N** -> repeated layers containing attention, feed-forward networks, normalization, and residual connections
- **LM Head** -> projects the final hidden state to vocabulary size and produces next-token probabilities

Inside each transformer block, the most important pieces are:

- **Multi-Head Self-Attention** -> each token computes relationships to other tokens using **Q**, **K**, and **V**
- **Feed-Forward Network (FFN)** -> processes each token representation after attention
- **Layer Norm + Residual Connections** -> stabilize training and help gradients flow cleanly

Detailed transformer mechanics are covered in the **Transformers** section below.

### Full Training Pipeline

LLM training is a three-phase pipeline:

- **Pretraining**
- **Supervised Fine-Tuning (SFT)**
- **Alignment** through approaches such as **RLHF**, **DPO**, or **Constitutional AI / RLAIF**

Each phase has a different objective, different data, and a different role in producing the final deployed model. A neural network starts with random weights and learns by repeatedly making predictions, measuring error, and adjusting weights to reduce that error. Training is the only phase where model weights are updated. Once deployed, the model is frozen and does not learn from user interactions.

```text
Raw Text Corpus
-> [Pretraining]
-> Base Model
-> [SFT]
-> Instruction-Following Model
-> [RLHF / DPO / RLAIF]
-> Aligned Assistant Model
```

End-to-end training flow:

```text
PHASE 1 - PRETRAINING
Raw Text Corpus
-> Data Cleaning / Deduplication / Quality Filtering
-> Tokenized Dataset
-> Forward Pass -> Cross-Entropy Loss -> Backpropagation -> AdamW Update
-> Repeat at massive scale across distributed GPUs
-> Base Model

PHASE 2 - SFT
Instruction-Response Pairs
-> Same next-token loss, usually only on response tokens
-> Full fine-tuning or LoRA / QLoRA
-> SFT Model

PHASE 3 - ALIGNMENT
SFT Model
-> RLHF or DPO or Constitutional AI / RLAIF
-> Aligned Assistant Model
```

#### Phase 1 - Pretraining

**Objective:** learn language, world knowledge, code patterns, and reasoning structures from raw text at massive scale. This is by far the most expensive phase and where most of the model's capability comes from.

**Task - Next-Token Prediction:** the model sees a token sequence and must predict the next token at each position. This is **self-supervised** because the training signal comes from the text itself; no human labels are needed. Given:

```text
"The cat sat on the"
```

the model should predict:

```text
"mat"
```

This simple objective becomes extremely powerful at trillion-token scale because the model must internalize grammar, syntax, facts, code patterns, reasoning-like patterns, and broad world regularities in order to make good predictions.

**Training data:** pretraining corpora usually include web text, books, Wikipedia, code, academic papers, forums, and curated high-quality sources. Data quality matters as much as scale. Raw internet text is noisy, so modern pipelines rely heavily on:

- deduplication
- toxicity filtering
- language identification
- document quality scoring
- source balancing

The ratio of web data, books, code, scientific text, and curated sources strongly shapes model behavior and capability.

**Scaling laws:** modern training follows the idea that for a fixed compute budget, model size and number of training tokens must be balanced carefully. This is the core insight from **Chinchilla scaling laws**.

**Forward pass:** the full pretraining forward pass looks like:

```text
Input Tokens
-> Embedding Layer
-> Transformer Blocks
-> LM Head
-> Logits
-> Softmax
-> Predicted Probability Distribution
```

**Loss function - Cross-Entropy**

```text
Loss = -log(P(actual_next_token))
```

If the model assigns high probability to the true next token, the loss is low. If the probability is low, the loss is high. This loss is computed across many token positions in parallel in every batch.

**Backpropagation:** gradients are computed for all model parameters using the chain rule. Those gradients flow backward from the **LM Head**, through all transformer blocks, and back to the embedding layer. Each gradient tells the optimizer how to adjust that weight to reduce loss.

**Optimizer - AdamW:** most modern LLMs are trained with **AdamW**, which maintains:

- a running average of gradients
- a running average of squared gradients
- weight decay for regularization

This makes training much more stable than plain SGD.

**Learning rate schedule:** training usually uses:

- **warmup** -> gradually increase learning rate early in training
- **decay** -> reduce learning rate later for stable convergence

Common practice is linear warmup followed by cosine decay.

**Gradient clipping:** prevents rare but catastrophic gradient spikes from destabilizing training.

**Mixed precision training:** training at scale usually uses **BF16** for efficiency, while keeping some optimizer states in higher precision for stability.

**Distributed training:** frontier-scale models do not fit on a single GPU. Modern training combines multiple strategies:

- **Data Parallelism** -> replicate the model, split the batch across GPUs, then average gradients
- **Tensor Parallelism** -> shard large weight matrices across GPUs
- **Pipeline Parallelism** -> split transformer layers across devices
- **ZeRO / optimizer sharding** -> shard optimizer states, gradients, and parameters to reduce redundancy

This is how training scales to thousands of GPUs and trillions of tokens.

The result of pretraining is a **base model**. It knows language, facts, code, and many reasoning patterns, but it does not yet know how to behave like a helpful assistant.

#### Phase 2 - Supervised Fine-Tuning (SFT)

**Problem SFT solves:** a pretrained model is just a text completer. It can continue text, but it does not inherently know how to follow instructions, act like an assistant, or format answers helpfully.

**Objective and data:** the model is fine-tuned on high-quality human-written instruction-response pairs. The learning objective is still next-token prediction, but the loss is usually applied only to the response tokens while the instruction tokens are masked.

This teaches the model:

- instruction following
- assistant-style formatting
- conversational response patterns
- better task behavior

Quality matters more than raw quantity. A relatively small but carefully curated SFT dataset can shift behavior dramatically.

**Full Fine-Tuning vs PEFT**

**Full fine-tuning** updates all weights. It is expensive and increases the risk of catastrophic forgetting.

**Parameter-Efficient Fine-Tuning (PEFT)** keeps the base model mostly frozen and updates only a small subset of parameters.

Important PEFT methods:

- **LoRA** -> learn low-rank update matrices instead of updating full weight matrices
- **QLoRA** -> apply LoRA to a quantized base model, making large-model fine-tuning much cheaper
- **Prefix Tuning / Prompt Tuning** -> learn soft prompt vectors prepended to the input

**Catastrophic forgetting:** if you fine-tune too aggressively on a narrow dataset, the model can lose broad pretrained capability. Common mitigations include:

- lower learning rates
- early stopping
- LoRA / QLoRA
- careful dataset design

The output of this stage is an **SFT model**. It follows instructions, but it is not yet fully aligned for safety, honesty, or human preferences.

#### Phase 3 - Alignment

**Problem alignment solves:** SFT teaches the model to respond like an assistant, but it does not fully ensure helpfulness, honesty, harmlessness, calibrated uncertainty, or human-preferred behavior. Alignment is the stage that shapes those properties.

#### RLHF

**RLHF (Reinforcement Learning from Human Feedback)** works like this:

```text
SFT model generates multiple responses
-> Human raters rank them
-> Reward Model is trained on the rankings
-> PPO fine-tunes the LLM to maximize reward
-> Aligned model
```

Key pieces:

- **Reward Model** -> predicts which outputs humans prefer
- **PPO** -> reinforcement learning method used to push the model toward higher reward
- **KL divergence penalty** -> keeps the model from drifting too far from the SFT policy and helps prevent reward hacking

RLHF was foundational for early frontier assistants, but it is expensive, labor-intensive, and can be unstable.

#### DPO

**DPO (Direct Preference Optimization)** is a simpler alternative. Instead of training a reward model and then running PPO, DPO directly trains the LLM on preference pairs:

```text
Prompt + Chosen Response + Rejected Response
-> Direct loss on the LLM
-> Aligned model
```

Advantages:

- simpler pipeline
- no separate reward model
- no RL loop
- often more stable in practice

This is why DPO is increasingly popular in modern open and production alignment pipelines.

#### Constitutional AI / RLAIF

**Constitutional AI** and **RLAIF (Reinforcement Learning from AI Feedback)** aim to scale alignment without depending entirely on large human rater teams.

Typical flow:

```text
LLM generates response
-> LLM critiques response using constitutional principles
-> LLM revises response
-> Revised outputs are used for SFT-style training
-> AI-generated rankings are used for preference optimization
-> Aligned model
```

This makes alignment more scalable and more explicitly principle-driven.

#### Key Interview Concepts

- **Reward hacking** -> the model finds ways to maximize reward without actually improving quality
- **Catastrophic forgetting** -> narrow fine-tuning causes loss of broad pretrained capability
- **Stability-plasticity dilemma** -> the more adaptable a model is to new data, the more likely it is to forget older knowledge
- **Data quality > data quantity** -> better filtered data often beats larger but noisier data
- **Chinchilla scaling laws** -> good performance depends on balancing model size and training tokens, not just maximizing parameters

#### Training Pipeline Real-World Usage

- **InstructGPT / ChatGPT-style systems** -> classic example of SFT plus RLHF producing assistant behavior
- **LLaMA-family fine-tuning** -> LoRA and QLoRA made domain adaptation practical even on much smaller hardware
- **Claude-style alignment** -> Constitutional AI and RLAIF reduce dependence on large human ranking teams
- **Enterprise adaptation** -> organizations often start from a base or instruction-tuned model and use PEFT methods for domain-specific specialization

### Inference

Inference is the process of using a trained, frozen LLM to generate responses. Training happens once; inference happens continuously in production and is what actually drives most real-world serving cost. The model weights do not change during inference. Every API call to OpenAI, Anthropic, Gemini, a self-hosted vLLM server, or a local model runner is an inference call.

From an interview and systems-design perspective, inference is one of the most important practical topics because most application engineers work far more with **serving**, **latency**, **throughput**, and **cost optimization** than with model training itself.

#### End-to-End Inference Pipeline

```text
User Request
-> API Gateway / Load Balancer
-> Inference Server Queue (vLLM / TGI style serving)
-> Full Context Assembly
   [System Prompt] + [History] + [RAG Chunks] + [User Query]
-> PREFILL
   all input tokens processed in parallel
   KV Cache populated
   first output token generated
-> first token streamed to user
-> DECODE LOOP
   one token at a time using KV Cache reuse
-> stop token or max_tokens reached
-> final response complete
```

#### The Two Phases of Inference

Every inference request has two computationally different phases:

##### Prefill

During **prefill**, the full known input context is processed in one parallel forward pass.

This includes:

- system prompt
- conversation history
- RAG-retrieved chunks
- the latest user query

Because all these tokens are known upfront, the model can process them in parallel through all transformer layers. The output of this phase is:

- a populated **KV Cache**
- the first predicted output token

Important point:

- **prefill latency is driven mostly by input length**
- the key UX metric here is **TTFT (Time To First Token)**

##### Decode

During **decode**, the model generates output autoregressively, one token at a time.

At each step:

- the newest token is fed into the model
- the model reuses the **KV Cache**
- one more output token is produced
- that token can be streamed immediately

Important point:

- **decode latency is driven mostly by model size and number of output tokens**
- the key metric here is **TPOT (Time Per Output Token)**

Simple view:

```text
PREFILL
All input tokens -> processed in parallel
-> KV Cache + Token_1

DECODE
Token_1 -> Token_2
Token_2 -> Token_3
Token_3 -> Token_4
...
```

#### Key Inference Metrics

- **TTFT (Time To First Token)** -> how quickly the user sees the first streamed token
- **TPOT (Time Per Output Token)** -> streaming speed after generation begins
- **Throughput** -> total tokens served per second across concurrent users

#### KV Cache - Most Important Inference Optimization

During decode, self-attention needs access to the **Key** and **Value** matrices for all prior tokens. Without caching, those would be recomputed repeatedly, making generation far more expensive.

The **KV Cache** stores these matrices in memory so that at each decode step only the new token's K and V need to be computed.

Why it matters:

- makes autoregressive generation practical
- reduces repeated computation
- is one of the most important optimizations in modern serving

Tradeoff:

- the KV Cache can become extremely large for long contexts
- long context windows directly increase memory usage per request
- this reduces how many concurrent requests one GPU can serve
- for very large models and long contexts, KV cache memory can become comparable to or larger than the effective active model footprint during serving

**Prompt caching** is a related provider-level optimization. If the same prompt prefix appears repeatedly, providers can reuse cached computations for that prefix and reduce both latency and cost. In practice, this is why static content such as system prompts, few-shot examples, and reusable prompt prefixes should be placed at the very beginning of context and kept stable across requests.

#### Quantization - Running Models More Cheaply

Quantization reduces model precision so that weights use less memory and inference becomes cheaper and faster.

Useful rule of thumb:

- **BF16** uses about 2 bytes per parameter
- a 70B model in BF16 therefore needs roughly 140GB just for weights
- this is why quantization is often necessary for practical self-hosted inference

Common forms:

- **INT8** -> lower memory with very small quality loss; safe default for many production use cases
- **INT4 / 4-bit** -> much smaller memory footprint; useful when hardware is constrained, though some reasoning quality may drop
- **AWQ** -> stronger 4-bit quantization that preserves activation-sensitive weights better
- **GPTQ** -> post-training quantization optimized to minimize reconstruction error
- **GGUF / llama.cpp formats** -> optimized for local and consumer hardware inference
- **QLoRA** -> mainly a fine-tuning technique, but important to understand because it combines quantization with LoRA to make large models practical on smaller hardware

Rule of thumb:

- **INT8** is usually close to lossless
- **INT4** is often acceptable for many practical tasks
- going much lower than 4-bit usually causes larger quality degradation

#### Batching - Maximizing GPU Utilization

GPUs are most efficient when many requests are served together.

**Static batching** groups requests into one batch and processes them together, but it is inefficient because short requests may have to wait for long ones.

**Continuous batching** is the dominant modern approach. New requests join the running batch dynamically as soon as a slot is free, which reduces GPU idle time and improves throughput significantly.

```text
Continuous batching:
Slot 1: [Req A][decode][decode][done]
Slot 2: [Req B][decode][done]
Slot 3: [Req C][decode][decode][decode][done]
Slot 4:                [Req D][prefill][decode][decode]
```

This is one of the main reasons systems like **vLLM** and **TGI** are so effective for self-hosted serving.

#### Prefill-Decode Disaggregation

At very large scale, some systems separate prefill and decode onto different serving resources.

Why this helps:

- **prefill** is more compute-bound
- **decode** is more memory-bandwidth-bound
- different hardware setups may be better for one phase than the other

This is mainly a hyperscaler optimization, but it is useful to know because it shows that prefill and decode are not just logically different; they are also different from a systems-performance perspective.

#### PagedAttention and vLLM

Naive KV-cache allocation can waste large amounts of memory by reserving more space than a sequence actually needs.

**PagedAttention** fixes this by splitting KV cache storage into fixed-size blocks and allocating them on demand, similar to virtual memory paging in operating systems.

Why it matters:

- reduces memory fragmentation
- increases concurrent request capacity
- is a major reason **vLLM** became the dominant self-hosted inference server

#### Speculative Decoding

Speculative decoding uses a small fast draft model to guess several next tokens ahead, and then a larger target model verifies them in parallel.

If the large model agrees, multiple tokens are accepted at once. If not, generation falls back at the disagreement point.

Why it matters:

- improves inference speed substantially
- can give 2-3x speedups in the right setup
- preserves output quality when verification is done correctly

#### Inference-Time Scaling

Another important idea is **test-time compute** or **inference-time scaling**. Instead of always using a larger model, systems spend more compute during inference to improve answer quality.

Examples:

- **Chain-of-Thought (CoT)** -> generate intermediate reasoning steps
- **Self-Consistency** -> generate multiple candidate answers and vote
- **Best-of-N Sampling** -> generate N outputs and choose the best with a verifier or reward function
- **reasoning-first models** -> allocate more internal compute for hard problems before responding

This increases cost, but often improves reliability and accuracy on difficult tasks.

#### Hosted API vs Self-Hosted Inference

| Aspect | Hosted API | Self-Hosted |
| --- | --- | --- |
| Setup | Very easy | Significant infra work |
| Cost at low scale | Usually better | Usually worse |
| Cost at high scale | Can become very expensive | Can become cheaper |
| Data privacy | External provider | Full internal control |
| Model control | Limited | Full control and fine-tuning |
| Latency | Shared infra | More predictable if dedicated |
| Compliance | Harder in regulated industries | Easier to control directly |

At sufficiently high token volume, self-hosting often becomes economically attractive, especially when privacy or compliance also matters.

#### Real-World Usage

- **Hosted APIs at scale** -> rely on batching, prompt caching, and specialized serving infrastructure across large GPU fleets
- **OpenAI / Anthropic style serving** -> repeated prompt prefixes benefit from provider-side prompt caching and large-scale decode optimization
- **Groq-style serving** -> shows that decode is often limited more by memory bandwidth than pure compute
- **Ollama / local development** -> runs quantized models locally for privacy and zero per-call API cost
- **Streaming web apps** -> send decode-phase tokens to the browser as they are generated so users do not wait for the full response
- **Next.js / Vercel-style integrations** -> often stream tokens over SSE or web streams because non-streaming UX becomes poor for longer decode phases

#### Production Implications

- always stream responses in user-facing applications when possible
- keep prompt prefixes stable to maximize prompt-cache benefits
- monitor **TTFT** and **TPOT** separately because they have different bottlenecks
- long contexts increase KV-cache memory and reduce concurrency
- quantization is often essential for efficient self-hosted serving
- continuous batching is one of the biggest throughput optimizations in production
- at higher scale, self-hosting can become cheaper than API pricing depending on workload and hardware

### Mixture of Experts (MoE)

Some LLMs use **Mixture of Experts**. Instead of activating the full network for every token, a routing mechanism selects only a small subset of expert subnetworks for that token. This allows very large total model capacity while keeping each forward pass cheaper than activating every parameter.

### Emergent Capabilities

At sufficient scale, LLMs sometimes show capabilities that were not obvious in smaller models, such as:

- few-shot learning
- stronger code generation
- analogical behavior
- more complex multi-step reasoning

This is one reason the **scaling hypothesis** remains such a major debate.

### Key Limitations

- **Hallucination** -> fluent output can still be factually wrong
- **Knowledge cutoff** -> weights are frozen after training
- **No persistent memory** -> models are stateless across API calls unless memory is engineered externally
- **Context window limit** -> each request has a hard token budget
- **Prompt sensitivity** -> small wording changes can shift output significantly
- **No true causal understanding** -> strong pattern modeling is not the same as genuine reasoning

### Real-World Usage

- **Microsoft Copilot** -> injects the user's documents, emails, or workspace data into context on each call
- **Enterprise RAG systems** -> combine long context windows with retrieved documents for contract, report, or policy analysis
- **On-premise open-weight deployment** -> organizations use models like **LLaMA** locally for privacy, compliance, or data residency requirements
- **GitHub Copilot** -> packs file context, surrounding code, and cursor position into the prompt and applies next-token prediction over code tokens

| Model        | Params      | Training Tokens |
| ------------ | ----------- | --------------- |
| GPT-3        | 175B        | 300B            |
| LLaMA 3 8B   | 8B          | 15T             |
| LLaMA 3 70B  | 70B         | 15T             |
| GPT-4 (est.) | ~1.8T (MoE) | Undisclosed     |

## AI vs AGI

**AI (Artificial Intelligence)** in practice today refers to **Narrow AI**. These are systems that can perform specific tasks intelligently but cannot operate flexibly outside the domains they were trained for. Every widely deployed system today, including **GPT**, **Claude**, **Gemini**, **Midjourney**, and self-driving systems, still falls into this narrow category.

**AGI (Artificial General Intelligence)** is a hypothetical system that can understand, learn, and perform any intellectual task a human can, with similar flexibility across domains. **AGI does not exist today.**

### Capability Spectrum

One useful way to view the landscape is as a capability spectrum:

```text
Rule-based AI -> Machine Learning -> Deep Learning -> Foundation Models / LLMs <- we are here -> AGI -> ASI
```

A slightly more detailed view:

- **Rule-based AI** -> explicit rules, no learning
- **Machine Learning** -> learns patterns from data
- **Deep Learning** -> neural networks with hierarchical representations
- **Foundation Models / LLMs** -> large pretrained models with broad but still narrow capabilities
- **AGI** -> hypothetical human-level general intelligence across domains
- **ASI (Artificial Superintelligence)** -> hypothetical intelligence beyond humans across all domains

### Why Current LLMs Are Still Narrow AI

Why current **LLMs** are still definitively **Narrow AI**:

- **No persistent learning** -> model weights are fixed after training and do not update from your conversation
- **No true generalization across all domains** -> they can transfer within learned statistical patterns, but not with unrestricted human-like flexibility
- **No genuine causal reasoning** -> they predict likely sequences, not true cause-and-effect understanding
- **No grounded common sense** -> they do not have embodied experience of the world
- **No self-directed goals** -> they respond to prompts and do not maintain durable autonomous objectives by default
- **Hallucinations** -> they can generate plausible but incorrect outputs because statistical fluency is not the same as understanding

### What AGI Would Require

What **AGI** would require:

- cross-domain generalization without retraining
- continuous learning without catastrophic forgetting
- causal reasoning
- common sense and grounded world understanding
- self-directed goals and long-horizon planning
- metacognition, including awareness of uncertainty and knowledge limits

### Scaling Hypothesis and the AGI Debate

One major debate is the **scaling hypothesis**. The question is whether AGI will emerge by scaling current transformer-based systems with more **data**, **parameters**, and **compute**, or whether entirely new architectures are required. Supporters of scaling argue that larger systems continue to show surprising gains. Critics argue that transformers still lack persistent memory, grounding, and true causal reasoning, so scaling alone may not be enough.

Another reason this debate continues is that **LLMs show emergent capabilities**. At larger scales, they sometimes display abilities not obvious at smaller scales, such as stronger multi-step reasoning, code generation, or analogical behavior. Some people see this as evidence that we may be moving toward AGI. Others see it as more advanced pattern matching rather than real general intelligence.

One benchmark often discussed in this context is **ARC-AGI**, designed to test novel reasoning rather than memorization. It is frequently cited in arguments that current frontier models are still far from AGI-level reasoning despite strong performance on language-heavy benchmarks.

Different organizations also define AGI differently. A practical framing often associated with **OpenAI** is a system that can outperform humans at most economically valuable tasks. Other groups focus more on broader philosophical or cognitive definitions. This matters because whether a system is called AGI often depends as much on the definition as on the capability itself.

### Why This Matters for Production System Design

Why this matters for production system design:

- LLMs do not have persistent memory unless you engineer it
- memory must be built externally through conversation history, summaries, or retrieval systems
- outputs must be validated programmatically
- agentic systems are engineered approximations of autonomy built on top of fundamentally narrow, stateless models
- you should never assume the model "understands" context the way a human does

## Fine-Tuning

Fine-tuning is the process of taking a pretrained LLM and continuing to train it on a smaller, targeted dataset so the model adapts to a specific use case. The model already knows language, general reasoning, and broad world knowledge. Fine-tuning adjusts those existing weights so the model emphasizes the right behavior, vocabulary, format, tone, or domain patterns for a particular task. It does not teach the model entirely new reasoning abilities from scratch.

There are two independent dimensions in fine-tuning:

- **what you are teaching** -> the objective and data type, such as **SFT**, **continued pretraining**, **instruction tuning**, **RLHF**, or **DPO**
- **how much of the model you are changing** -> the efficiency dimension, such as **full fine-tuning**, **LoRA**, **QLoRA**, **prefix tuning**, **prompt tuning**, or **IA3**

These dimensions are independent. For example, **SFT** can be done with full fine-tuning or with **LoRA**.

### When to Fine-Tune vs Alternatives

Always try **prompt engineering** first. A strong system prompt plus a few-shot examples is cheaper, faster, and easier to update. Fine-tune only when prompting repeatedly fails to achieve the required behavior consistently.

Important rule:

- do **not** fine-tune just to add factual knowledge
- factual knowledge belongs in **RAG**
- fine-tuning is best for **behavior, style, tone, format, and domain adaptation**

The best production systems often combine both:

- **fine-tuning** for behavior and output format
- **RAG** for factual grounding

### Fine-Tuning Flow

```text
Pretrained Base Model
-> Choose Training Objective
   (SFT / Continued Pretraining / RLHF / DPO / Instruction Tuning)
-> Choose Efficiency Method
   (Full Fine-Tuning / LoRA / QLoRA / Prefix Tuning / Prompt Tuning / IA3)
-> Train on Targeted Data
-> Adapted Model
-> Optional RAG for factual grounding
```

### What You Are Teaching - Training Methods

#### Supervised Fine-Tuning (SFT)

**SFT** is the most common form of fine-tuning. The model is trained on:

- instruction
- ideal response

pairs using next-token prediction loss, usually computed only on the response tokens while the instruction tokens are masked.

The most important point here is that **data quality dominates**. A small set of highly consistent, high-quality examples often beats a much larger set of mediocre examples. Because the model already knows how to reason and write, SFT is mostly teaching a specific behavioral pattern, and noisy examples corrupt that signal quickly.

InstructGPT is the classic example here. It used only around **13,000** carefully curated SFT examples, yet that was enough to shift the behavior of a very large base model significantly. This is why interview answers should emphasize that in SFT, **quality beats quantity** much more than people expect.

Template correctness also matters. Chat-tuned models expect the correct conversation format, and using the wrong template can cause silent failure where training appears to run but the model does not learn the desired behavior.

Examples:

- **LLaMA-style templates** -> `[INST] ... [/INST]`
- **ChatML-style templates** -> `<|im_start|>user ... <|im_end|>`
- if the training data does not match the model's expected chat template, the model may train successfully but still fail to behave correctly at inference time

#### Continued Pretraining (Domain Adaptive Pretraining)

**Continued pretraining** uses raw domain text rather than instruction-response pairs. The model is further trained with the same next-token prediction objective used in original pretraining.

This is appropriate when the model does not understand the domain language well enough yet, such as:

- finance
- medicine
- law
- specialized code or scientific corpora

The idea is:

- first teach the model the domain language
- then fine-tune or align it for tasks inside that domain

This is the path used for domain-heavy systems such as **financial**, **medical**, and **scientific** LLMs. A common mental model is:

- **continued pretraining** teaches the model how the domain talks
- **SFT / instruction tuning** teaches the model how to perform tasks in that domain

#### Instruction Fine-Tuning

**Instruction fine-tuning** is SFT performed on a large and diverse set of instruction-following examples so the model becomes broadly better at following many kinds of tasks rather than specializing narrowly on one.

This is what usually turns a raw base model into an **Instruct** or **Chat** model.

#### RLHF

**RLHF (Reinforcement Learning from Human Feedback)** aligns the model to human preferences beyond what SFT alone can do.

Typical flow:

1. the SFT model generates multiple responses
2. human raters rank them
3. a reward model is trained on those rankings
4. the LLM is optimized with PPO to maximize reward while staying close to the SFT model via a **KL penalty**

This is powerful, but expensive and operationally complex.

The **KL penalty** matters because without it the model can start gaming the reward model instead of genuinely improving. This is the classic **reward hacking** problem.

#### DPO

**DPO (Direct Preference Optimization)** aims for the same preference-alignment goal as RLHF, but removes the separate reward model and RL loop.

Instead, it trains directly on:

- prompt
- chosen response
- rejected response

This makes alignment much simpler and more stable. It has become the default alignment approach for many modern open-source and practical production fine-tuning setups.

Interview angle:

- **RLHF** -> more complex pipeline, separate reward model, PPO
- **DPO** -> one-stage preference optimization, simpler and usually more stable
- both aim to align the model with human preference, but **DPO** is much easier for most teams to operate

### How Much You Change - Efficiency Methods

#### Full Fine-Tuning

**Full fine-tuning** updates every model parameter. It is the most flexible option, but also the most expensive. For large models, it often requires infrastructure similar to pretraining-scale setups.

#### LoRA

**LoRA (Low-Rank Adaptation)** is the most important PEFT method.

Core idea:

- the weight update does not need to use the full parameter space
- instead of updating the full matrix directly, LoRA learns two small low-rank matrices
- the original base weights remain frozen

Mathematically, the update is written as:

```text
Delta W = A x B
```

So instead of training the full weight matrix **W**, the model only learns the low-rank update **A x B**. This is why LoRA reduces trainable parameters so dramatically.

Benefits:

- far fewer trainable parameters
- much lower GPU memory usage
- adapters can be merged into the base weights
- adapters can also remain separate and be hot-swapped at inference time

Typical practical pattern:

- start with small rank values such as `r = 8` or `r = 16`
- apply LoRA first to attention projections such as **Q** and **V**
- expand to more modules only if the task needs more adaptation capacity

This is one of the main reasons LoRA is so practical for multi-tenant and multi-customer deployments.

#### QLoRA

**QLoRA** applies LoRA on top of a quantized 4-bit base model. The base model stays frozen in quantized form while the adapters are trained in higher precision.

Why it matters:

- dramatically reduces memory requirements
- enables large-model fine-tuning on consumer or modest GPUs
- made open-source LLM fine-tuning widely accessible

Important detail:

- QLoRA usually uses **NF4 4-bit quantization**, which is designed to work well for neural network weight distributions
- the base model is frozen and quantized, while the LoRA adapters are still trained in higher precision
- quality is often very close to standard LoRA, which is why QLoRA became so widely adopted

#### Prompt Tuning, Prefix Tuning, and IA3

Other PEFT methods include:

- **Prompt Tuning** -> learn soft prompt vectors prepended to the input
- **Prefix Tuning** -> inject learned prefix vectors at transformer layers
- **IA3** -> learn scaling vectors applied to internal activations

These are lighter than LoRA, but often less expressive for difficult adaptation tasks.

Use them when:

- adaptation needs are relatively modest
- memory efficiency matters more than maximum flexibility
- LoRA would be unnecessary overkill for the task

### Catastrophic Forgetting

One of the most important fine-tuning risks is **catastrophic forgetting**. If the model is trained too aggressively on a narrow dataset, it may become better on the target task while losing general capability.

Typical symptoms:

- strong performance on the fine-tuning task
- unexpected failure on nearby tasks the base model previously handled well

Common mitigations:

- **LoRA / QLoRA**
- low learning rates
- early stopping
- mixing in more general data
- careful dataset design

Useful interview detail:

- full fine-tuning often uses very small learning rates such as `1e-5` to `1e-4`
- LoRA commonly tolerates somewhat higher learning rates such as `1e-4` to `3e-4`
- one of the safest strategies is to mix some general instruction data with the domain data so the model does not over-specialize

### Fine-Tuning Data Quality

Data quality is often the biggest determinant of fine-tuning success, more important than model size or long training runs.

The key idea is that the model already knows language and broad reasoning. Fine-tuning is mostly teaching **behavioral patterns**, so inconsistent or low-quality examples create a very noisy signal. A few hundred highly consistent examples can outperform tens of thousands of mediocre ones.

Key principles:

- **diversity** -> cover the real range of user inputs
- **consistency** -> keep output style and format stable
- **low ambiguity** -> examples should have a clearly correct target behavior
- **balanced difficulty** -> include easy, medium, and hard cases

Practical quantity guidance:

- behavior or tone adjustment -> often `100-1000` high-quality examples can be enough
- task-specific fine-tuning -> often `1,000-10,000`
- stronger domain adaptation -> often `10,000-100,000+`

Synthetic data generation with a strong LLM is now common, but it still needs filtering and quality control.

Best practice is:

- generate synthetic examples with a strong model
- filter them for correctness and formatting
- keep only examples that match the exact behavior you want the model to learn

### Decision Framework

| Situation | Best Approach |
| --- | --- |
| Need current or specific factual knowledge | RAG |
| Need source attribution in responses | RAG |
| Task already works with prompting | Prompting only |
| Need highly consistent output format or schema | SFT |
| Need domain vocabulary or tone | SFT or continued pretraining |
| Need to reduce prompt length at scale | SFT |
| Need alignment to human preferences | DPO or RLHF |
| Limited GPU memory | LoRA or QLoRA |
| Need many customized variants of one base model | LoRA with adapter swapping |
| Need both behavior change and factual grounding | Fine-tuning + RAG |

### Real-World Usage

- **BloombergGPT-style domain adaptation** -> continued pretraining on financial text before downstream task adaptation
- **ChatGPT-style alignment** -> GPT-3.5 style base model -> SFT -> RLHF with human raters, reward model, and PPO
- **LLaMA Instruct-style models** -> instruction tuning plus modern preference alignment such as DPO
- **Enterprise assistants** -> fine-tune output format, tone, and citation style, then combine with **RAG** for factual grounding
- **Multi-tenant SaaS** -> one base model with many hot-swappable **LoRA** adapters for customer-specific behavior
- **Morgan Stanley-style internal assistants** -> fine-tune response behavior for enterprise usage, but still rely on retrieval over proprietary documents for current factual answers

## Tokens

A **token** is the smallest unit of text that an AI model reads and processes. It's not exactly a word - it's a chunk of characters. AI models don't read text like humans. They convert text into numbers first. **Tokens** are that conversion unit - text gets split into tokens, tokens get converted to numbers, and the model works on those numbers. Think of tokens like Lego bricks: a sentence is a Lego structure, and before the model can work with it, it breaks the structure into individual bricks (**tokens**).

### Why Tokens Matter

In interviews, the most important thing to remember is:

- **Tokens are the actual units the model consumes, not words**
- **Pricing, latency, and context limits are all based on token count**
- A short-looking sentence can still become many tokens depending on punctuation, spacing, formatting, or code
- One word can also become multiple tokens, especially with **subword tokenization**

Modern LLMs usually rely on **subword tokenization** methods such as **Byte Pair Encoding (BPE)** or closely related variants. The broad idea is that common patterns are stored as reusable pieces, so the model can represent text efficiently without needing a separate token for every possible word.

### Tokens: Why, Where, How, When

- **Why** -> because the model cannot process raw text directly; it needs a numerical representation
- **How** -> text is split by a tokenizer into tokens, tokens are mapped to IDs, and IDs are converted into embeddings
- **Where** -> in every prompt, completion, API call, cost calculation, and context window limit
- **When** -> whenever text enters or leaves an LLM pipeline

### Tokenization Pipeline

Example: Tokenization Process

```text
Raw Text: "Hello, how are you?"
        ->
    Tokenizer (BPE)
        ->
Tokens: ["Hello", ",", " how", " are", " you", "?"]
        ->
Token IDs: [15496, 11, 703, 389, 345, 30]
        ->
    LLM Model Processes
        ->
Output Token IDs: [40, 1101, 1804, 11, 5678]
        ->
    Decoded back to text
        ->
Output: "I am fine, thanks!"
```

Steps in the tokenization pipeline:

1. **Raw Text** -> Human-readable input given to the model.
2. **Tokenizer** -> Splits text into subword tokens using learned patterns. Tokenization algorithm used by most modern LLMs (GPT, Claude, Gemini) is **Byte Pair Encoding (BPE)**.
3. **Tokens** -> Smaller text units that the model can understand.
4. **Token IDs** -> Each token is mapped to a unique numeric ID.
5. **Embeddings** -> Token IDs are converted into dense vectors capturing meaning.
6. **Model Processing** -> Transformer layers analyze context using attention mechanisms.
7. **Output Token IDs** -> Model predicts next tokens as probability-based IDs.
8. **Decoder** -> Converts predicted token IDs back into words.
9. **Output Text** -> Final human-readable response generated by the model.

One very useful interview line is: `Text -> Tokens -> Token IDs -> Embeddings -> Transformer -> Output IDs -> Text`.

## Embeddings

**Embeddings** are dense vectors of floating point numbers that represent the meaning of text in a mathematical space. Semantically similar text ends up close together in that space, while dissimilar text ends up farther apart. This turns language understanding into a geometry problem, which is why embeddings are so useful for search, clustering, recommendation, and retrieval.

### Why Embeddings Matter

Why embeddings matter:

- **Embeddings** convert symbolic token IDs or text into dense numerical vectors
- these vectors capture **semantic** and **contextual** relationships
- without embeddings, token IDs are just identifiers, not meaningful representations
- embeddings let systems compare meaning mathematically instead of relying only on exact keyword match

### Sparse vs Dense Representations

Before embeddings, text was often represented using sparse methods such as **one-hot encoding** or **TF-IDF**. Those methods create very large vectors with little semantic understanding. For example, "cat" and "feline" would be treated as unrelated features even though they are close in meaning. Embeddings solve this by learning compact representations where semantic relationships are reflected geometrically.

### How Embeddings Evolved

**Word2Vec** was the first major breakthrough in this direction. It learned word vectors by predicting surrounding words. This produced famous vector arithmetic examples such as:

`king - man + woman ~= queen`

Its limitation was that each word had one fixed vector regardless of context. So "bank" in the sense of finance and "bank" in the sense of a river shared the same representation. **Contextual embeddings** in transformer models solved this by generating a word representation based on the full surrounding sentence.

Modern embedding models are usually trained with **contrastive learning**. Similar text pairs are pushed closer together and dissimilar pairs are pushed farther apart. Architectures often use **Siamese encoders**, and large batches help because they provide more negative examples for separation.

Examples of modern embedding models often discussed in practice include:

- OpenAI `text-embedding-3-small` -> 1536 dimensions
- OpenAI `text-embedding-3-large` -> 3072 dimensions
- Cohere `embed-english-v3.0`
- `bge-large-en-v1.5`
- `all-MiniLM-L6-v2` -> 384 dimensions
- `nomic-embed-text`

### Similarity Metrics

How similarity is measured:

- **Cosine similarity** -> standard choice for text; measures angle between vectors; range `-1 to 1`; ignores magnitude  
  `cosine(A, B) = (A . B) / (|A| x |B|)`
- **Dot product** -> equivalent to cosine similarity when vectors are normalized
- **Euclidean distance (L2)** -> physical distance in vector space; less common for text because it is sensitive to magnitude

Important point:

- always check the embedding model's recommended similarity metric
- models are usually trained with a particular metric assumption

### Chunking Before Embedding

**Chunking** is the most important practical step before embedding long documents. Since embedding models have token limits, long documents must be split into chunks before indexing. For example, OpenAI embedding models commonly support up to around `8191` tokens for a single embedding input.

Common chunking strategies:

- **Fixed-size chunking** -> split every N tokens with overlap; simple but may cut across meaning boundaries
- **Sentence-level chunking** -> cleaner semantically because chunks break at sentence boundaries
- **Recursive chunking** -> split paragraph, then sentence, then smaller units while trying to preserve semantic units; a common example is LangChain's `RecursiveCharacterTextSplitter`
- **Semantic chunking** -> detect topic boundaries by measuring drops in cosine similarity between adjacent sentences; usually most accurate but most expensive

Typical practical guidance:

- **256 - 512 tokens** per chunk is a common sweet spot
- **10 - 20% overlap** often works well
- chunks that are too large hurt retrieval precision
- chunks that are too small lose important context

### Embeddings in a RAG Pipeline

RAG pipeline with embeddings:

```text
INDEXING:
Document -> Chunking -> Embedding Model -> Vector + Text + Metadata -> Vector DB

QUERYING:
User Query -> Embedding Model -> Query Vector
-> Vector DB Search
-> Top-K Similar Chunks
-> Inject into LLM Context
-> Grounded Response
```

Related concept: embeddings are also used outside generation. In retrieval systems, embeddings are used for:

- **semantic search**
- **document matching**
- **clustering**
- **recommendation**

In simple terms:

- embeddings help find relevant information
- the generation model uses that information to produce an answer

### Embedding Drift and Versioning

**Embedding drift / model versioning** is an important production concern. If you change the embedding model, old vectors and new vectors usually become incompatible because they live in different vector spaces. In practice, changing embedding models often requires re-embedding the full corpus, so this should be treated like a migration, not a trivial upgrade.

Other practical notes:

- some embedding models support requesting fewer dimensions to reduce storage and ANN query cost
- multimodal embeddings map text, image, audio, or other modalities into a shared vector space; examples include **CLIP** and **ImageBind**
- always batch embedding calls for efficiency instead of embedding one chunk at a time

### Real-World Usage

- **Notion-style semantic search** -> pages and notes are embedded so meaning-based search works even without exact keyword overlap
- **Spotify-style recommendations** -> songs or users can be embedded into a shared space for recommendation
- **GitHub Copilot-style code retrieval** -> semantically similar code can be matched even when syntax differs
- **Customer Support AI** -> past tickets and resolutions are embedded so new issues can retrieve relevant prior answers

### Production Implications

- always use the **same embedding model** for indexing and querying
- normalize embeddings when the retrieval setup expects cosine or dot-product similarity
- model upgrades may require **full corpus re-embedding**
- use multilingual embedding models for multilingual apps
- batch embedding requests for efficiency
- monitor embedding latency separately from LLM latency in RAG systems
- open-source embedding models can be run locally for privacy, cost, or compliance reasons

### Embeddings: Why, Where, How, When

- **Why** -> to represent tokens or documents in a form that captures meaning and similarity
- **How** -> tokens or text are converted into dense vectors in high-dimensional space
- **Where** -> inside LLM processing, semantic search, vector databases, recommendation systems, and retrieval pipelines
- **When** -> when the system needs similarity matching, retrieval, clustering, or semantic understanding

## Vector Databases

A **vector database** is a database purpose-built to store, index, and search high-dimensional vectors (**embeddings**) at scale. Unlike traditional databases that find records by exact value matching, vector databases find records by **semantic similarity** — returning vectors mathematically closest to a query vector. This operation is called **Approximate Nearest Neighbor (ANN) search** and is the core primitive that makes production **RAG** systems, semantic search, and embedding-based recommendation viable.

Traditional databases fail for vector search for two reasons:

- SQL keyword matching such as `LIKE '%term%'` misses semantically equivalent content written with different words
- brute-force cosine similarity against every stored vector is `O(n)` and becomes intractable at scale

The deeper reason naive spatial indexing fails is the **curse of dimensionality**. In high-dimensional spaces such as `768-3072` dimensions, all vectors become roughly equidistant, so tree-based spatial indexes like k-d trees and R-trees degrade toward brute-force performance. ANN algorithms solve this by exploiting the structure and connectivity of the data rather than relying on naive geometric partitioning.

### Indexing and Retrieval Pipeline

```text
INDEXING (offline, once):
Raw Documents
-> Chunking (256-512 tokens, 10-20% overlap, semantic boundaries)
-> Embedding Model
-> Dense Vectors + Metadata (text, source, user_id, timestamp)
-> Vector DB builds ANN index (HNSW in RAM / IVF on disk)

QUERYING (real-time, per request):
User Query
-> Same Embedding Model
-> Query Vector
-> ANN Search (HNSW traversal / IVF cluster search)
   + Metadata Filter applied inline
-> Top-50 Dense Candidates
-> BM25 Sparse Retrieval in parallel
-> Top-50 Sparse Candidates
-> RRF Fusion
-> Cross-Encoder Re-ranker
-> Top-5 Final Chunks
-> Original Text injected into LLM Context
-> Grounded Response
```

Critical rule:

- always use the **exact same embedding model** for indexing and querying
- vectors from different models live in different geometric spaces
- switching embedding models requires re-embedding the corpus and rebuilding the index

### Core Indexing Algorithms

#### HNSW - Hierarchical Navigable Small World

Think of **HNSW** like Google Maps navigation. You do not start a long trip by driving every local street. You first use highways to cover distance quickly, then smaller roads to get precise. HNSW builds exactly that kind of layered structure for vector search.

HNSW builds a multi-layer graph where each vector is a node:

- **Layer 0** contains all vectors
- higher layers contain progressively fewer nodes
- upper layers act like highways for long-distance traversal
- lower layers handle local precise navigation

Insertion:

- a new vector is assigned a maximum layer probabilistically
- at each occupied layer, it connects to its nearest neighbors
- weaker edges are pruned to preserve graph quality

Search:

- start at the entry point of the top layer
- greedily move to the neighbor closest to the query
- descend when no closer neighbor exists
- at layer 0, perform a beam search with width `ef_search`
- return the final top-K results

**Example - Support Article Search:** imagine 5 million support articles indexed with HNSW. Top layers quickly route the query into the right topic area, and the bottom layer finds the most relevant local articles. This is why HNSW can search huge corpora in milliseconds instead of tens of seconds.

Important parameters:

- **M** -> number of bidirectional connections per node; higher means better recall but more RAM
- **ef_construction** -> beam width during index build; higher means better graph quality but slower build
- **ef_search** -> beam width during query time; tunable per query, higher for recall and lower for speed

Complexity:

- search is roughly `O(log N)`
- build is roughly `O(N log N)`

Primary weakness:

- the graph must reside in RAM
- at very large scale, memory usage becomes the main bottleneck

Common in **Pinecone**, **Weaviate**, **Qdrant**, **Chroma**, **pgvector**, and **Redis VSS**.

HNSW intuition:

```text
Layer 2: A ---------------- B
Layer 1: A --- C --- D --- B --- E
Layer 0: A-F-G-C-H-I-D-J-K-B-L-M-E-N-O

Query: enter top layer
     -> greedy traverse toward query
     -> descend layers
     -> beam search at bottom layer
     -> return top-K
```

#### IVF - Inverted File Index

Think of **IVF** like a library organized into sections. Instead of searching every book, you first identify which sections are most relevant, then search only within those sections.

IVF works by:

- clustering all vectors into `K` groups using k-means
- assigning each vector to its nearest centroid
- storing an inverted list from centroid to member vectors

Search works like this:

- compare the query vector to all centroids
- select the nearest `nprobe` centroids
- search only inside those clusters
- return the top-K results

**Example - E-commerce Search:** if 2 million products are clustered into 1,000 groups, a query such as "wireless noise-cancelling headphones" only searches the most relevant nearby clusters rather than all products.

Important parameters:

- **nlist** -> number of clusters
- **nprobe** -> how many clusters to search at query time

Tradeoff:

- more clusters = faster cluster-level search but more boundary misses
- higher `nprobe` = better recall but slower queries

Key weakness:

- true nearest neighbors can be missed if they fall into non-searched clusters
- this is the core **cluster boundary problem**

Main advantage over HNSW:

- more memory-efficient
- vectors can live on disk more easily
- useful at much larger scale

#### IVF + PQ - Product Quantization

**IVF + PQ** combines cluster-based narrowing with aggressive vector compression.

Product Quantization works by:

- splitting a vector into multiple subvectors
- learning a codebook for each subvector position
- replacing each subvector with a compact codebook index

This can shrink a large float32 vector dramatically and make distance computation much faster via lookup tables instead of full floating-point operations.

**Example - Billion-Scale Search:** at very large scale, storing raw vectors may require terabytes of memory. IVF + PQ reduces that drastically, making billion-scale search practical where full-fidelity HNSW would be too expensive.

Tradeoff:

- strong memory savings
- lower recall than HNSW
- best used when memory constraints dominate

#### ScaNN - Scalable Approximate Nearest Neighbor

**ScaNN** is Google's production ANN algorithm used in systems like Search and YouTube. Its key idea is **anisotropic quantization**: not all quantization errors matter equally. Errors aligned with the query direction affect ranking much more than errors in irrelevant directions.

This makes ScaNN especially strong at preserving retrieval quality under compression. It is important conceptually even though it is less commonly surfaced directly in third-party vector database products than HNSW.

### Metadata Filtering

Production queries almost always combine semantic similarity with metadata constraints.

Two naive approaches both fail:

- **post-filtering** -> run ANN search first, then filter; relevant results may be lost if most candidates are filtered out
- **pre-filtering** -> filter first, then search only the subset; this can degrade toward brute force

**Example - Multi-tenant SaaS Knowledge Base:** if a global ANN search returns mostly documents from the wrong customer, a naive post-filter wastes the best candidates. If you filter first and then search a narrow slice inefficiently, you lose ANN efficiency. Modern vector databases solve this by integrating filtering directly into the ANN traversal itself.

### Hybrid Search - Dense + Sparse

Pure semantic search misses exact keyword matches that genuinely matter:

- product IDs
- acronyms
- legal terms
- version numbers
- proper nouns

Pure sparse retrieval such as **BM25** has the opposite weakness: it matches exact words but misses meaning.

Production systems therefore combine both:

- **Dense retrieval** -> ANN over embeddings for semantic similarity
- **Sparse retrieval** -> BM25-style lexical matching for exact terms
- **Fusion** -> combine dense and sparse rankings, often with **Reciprocal Rank Fusion (RRF)**

**RRF** works by combining reciprocal ranks such as:

`1 / (rank + k)`

across both ranked lists, so documents that rank well in both dense and sparse retrieval are strongly favored.

This is the recommended production default for many retrieval systems.

### Hybrid Retrieval Flow

```text
User Query
-> Dense Embedding Model -> Query Vector
-> Sparse Retrieval / BM25 Query

Dense Retrieval Path:
Query Vector -> ANN Search -> Dense Candidates

Sparse Retrieval Path:
Keyword Query -> BM25 / Sparse Search -> Sparse Candidates

Fusion:
Dense Candidates + Sparse Candidates
-> Reciprocal Rank Fusion (RRF) or learned fusion
-> Top-N Combined Candidates
-> Cross-Encoder Re-ranker
-> Top-K Final Chunks
-> Inject into LLM Context
-> Grounded Response
```

### Re-Ranking - Precision Layer

ANN gets you into the right neighborhood quickly. **Re-ranking** finds the best results inside that neighborhood precisely.

**Bi-encoder retrieval stage:**

- query and documents are encoded separately
- similarity is computed via vector comparison
- fast because document embeddings are precomputed
- less precise because query and document never interact during encoding

**Cross-encoder re-rank stage:**

- query and candidate chunk are processed together in one transformer
- full attention across both inputs produces much better relevance estimates
- much slower, so it is only used on a small shortlist

Typical production pattern:

```text
ANN retrieves top-50 candidates
-> Cross-encoder re-ranks all 50
-> Top-5 injected into LLM context
```

This is one of the highest-ROI improvements to production RAG quality.

### Chunking Strategy - Retrieval Quality Ceiling

Retrieval quality is fundamentally bounded by chunking quality. No ANN algorithm can recover semantic structure that was destroyed during poor chunking.

Key failure modes:

- **too large** -> chunks mix multiple topics and retrieval becomes noisy
- **too small** -> chunks lack enough standalone meaning
- **no overlap** -> meaning that spans chunk boundaries is lost

Important strategies:

- **fixed-size chunking** -> simple and predictable, but can split mid-sentence
- **recursive chunking** -> split by paragraph, then sentence, then smaller units; strong practical default
- **semantic chunking** -> split where similarity between adjacent sentences drops sharply
- **parent-child chunking** -> index small chunks but inject larger parent context when retrieved

Good default range:

- **256-512 tokens**
- **10-20% overlap**

### Vector DB Landscape

| Database | Type | Algorithm | Best For | Key Highlights |
| --- | --- | --- | --- | --- |
| **Pinecone** | Managed cloud | HNSW + proprietary | Production RAG | Fully managed, serverless tier, native hybrid search |
| **Weaviate** | OSS / cloud | HNSW | Hybrid search, multi-modal | GraphQL API, native BM25 hybrid, rich module ecosystem |
| **Qdrant** | OSS / cloud | HNSW | Complex filtering | Rust-based, strong payload indexing, sparse vector support |
| **Chroma** | OSS | HNSW | Local dev / prototyping | Simple setup, common LangChain default |
| **Milvus** | OSS / cloud | IVF+PQ, HNSW | Billion-scale | Large-scale open-source option |
| **pgvector** | PostgreSQL extension | HNSW, IVF | Existing Postgres infra | No new infra, SQL joins with vectors |
| **Redis VSS** | Redis extension | HNSW | Ultra-low latency | Good fit for Redis-heavy stacks |
| **FAISS** | Library (Meta) | IVF+PQ, HNSW | Research / custom builds | In-process library, not a DB |
| **Elasticsearch** | OSS / cloud | HNSW | Existing ES infra + hybrid | Native dense + BM25 hybrid |

**pgvector** deserves special attention for full-stack teams because it adds vector search directly inside PostgreSQL with minimal extra infrastructure. It works well early, and teams often migrate to a dedicated vector database only once latency or throughput becomes the bottleneck.

### Embedding Drift - The Migration Problem

When switching embedding models, every existing vector becomes incompatible because different models produce vectors in different spaces. That means:

- full corpus re-embedding
- full index rebuild
- retrieval validation before cutover
- version-aware rollout and rollback planning

Treat embedding model changes like real database migrations, not simple upgrades.

### Production Implications

- always use the same embedding model for indexing and querying
- implement hybrid search by default for production retrieval
- add a cross-encoder re-ranker between ANN retrieval and LLM context injection
- normalize vectors when the chosen similarity setup expects it
- tune runtime parameters such as **ef_search** or **nprobe** without rebuilding the index
- monitor retrieval quality separately from LLM output quality
- start with simpler infrastructure such as **pgvector** when it is sufficient
- plan embedding-model upgrades as full re-embedding migrations
- remember that vector retrieval and LLM prompt caching are separate optimization layers

### Real-World Usage

- **Notion-style workspace search** -> semantic retrieval with workspace-level metadata filtering
- **E-commerce search** -> combine dense retrieval with BM25 for product discovery
- **LangChain / LlamaIndex pipelines** -> embeddings, vector retrieval, re-ranking, and LLM grounding
- **Fraud or anomaly retrieval** -> compare new events against semantically similar historical patterns

## Transformers

**Transformers** are a type of deep learning model designed to process sequential data using **attention** instead of recurrence like **RNNs** or convolution like **CNNs**. They were introduced in the paper **"Attention Is All You Need"** and became the backbone of modern AI systems such as **GPT**, **BERT**, **ViT**, and many multimodal models.

The core idea is simple:

- instead of processing tokens one by one
- transformers look at all tokens together
- then learn relationships between them using **attention**

This is why transformers are so effective for language: they can capture relationships between words that are far apart in a sequence without relying on long chains of hidden states.

### Transformers: Why, Where, How, When

- **Why** -> to model relationships in sequences more effectively than older architectures
- **How** -> by using **self-attention**, **multi-head attention**, **feed-forward layers**, and **positional information**
- **Where** -> LLMs, machine translation, summarization, vision transformers, audio models, multimodal systems
- **When** -> when the task involves long-range dependencies, large-scale sequence modeling, or high-performance parallel training

Transformers are still **neural networks**. They are built from layers, trained with **backpropagation**, optimized with **gradient descent**, and use familiar concepts like **weights**, **biases**, and **loss functions**.

### How Transformers Differ from Traditional Neural Networks

#### Similarities

- built using layers of neurons
- trained using backpropagation and gradient descent
- learn patterns from data
- use weights, biases, and loss functions

#### Differences

- **Processing**
  - **Transformers** -> process tokens in parallel
  - **RNNs** -> process tokens sequentially
- **Context handling**
  - **Transformers** -> capture long-range dependencies using attention
  - **RNNs** -> struggle with long-term dependencies because of vanishing gradients
- **Core mechanism**
  - **Transformers** -> attention with **Query (Q)**, **Key (K)**, and **Value (V)**
  - **CNNs** -> convolution filters
  - **RNNs** -> hidden states
- **Performance**
  - **Transformers** -> faster on GPUs because they are parallelizable
  - **RNNs** -> slower because they are sequential
- **Complexity**
  - **Transformers** -> attention is typically `O(n^2)`
  - **RNNs** -> sequential processing is typically `O(n)`

### High-Level Transformer Architecture

- **Input Embedding**
- **Positional Encoding**
- **Self-Attention**
- **Feed Forward Network**
- **Layer Normalization**
- **Residual Connections**

### How Transformers Work Step by Step

1. **Input tokens** are converted into **embeddings**
2. **Positional encoding** is added so the model knows token order
3. A **self-attention** layer computes relationships between tokens
4. **Multi-head attention** learns multiple types of relationships in parallel
5. A **feed-forward network** processes each token representation
6. **Residual connections** and **layer normalization** stabilize training
7. Multiple transformer blocks are stacked
8. The output layer produces **logits**, which are then converted to probabilities

### Encoder vs Decoder

- **Encoder** (example: **BERT**)
  - reads the full input
  - uses **bidirectional attention**
- **Decoder** (example: **GPT**)
  - generates tokens one by one
  - uses **causal attention**, meaning it can only attend to previous tokens

### Simplified Flow

```text
Input Text
    ->
Tokenization
    ->
Embeddings + Positional Encoding
    ->
[Self-Attention -> Feed Forward] x N
    ->
Output (logits -> softmax)
```

### Why Transformers Are Powerful

- they capture **global context**
- they train efficiently using **parallel computation**
- they scale well with more **data** and **compute**
- they form the backbone of modern **LLMs**

### Common Interview Questions

- What problem do transformers solve over RNNs?
- What is **self-attention**?
- Why is **positional encoding** needed?
- What is **multi-head attention**?
- What is the difference between **encoder** and **decoder**?

### Common Mistakes

- saying transformers are not neural networks
- ignoring **positional encoding**
- not explaining **attention** clearly
- confusing **encoder** and **decoder** roles

### One-Liner

Transformers are neural network architectures that use **attention** to process tokens in parallel and capture relationships in data efficiently.

### Memory Tricks

- **Transformer** = attention-based neural network
- **RNN** = sequence memory, **Transformer** = full-context view

## Context

### What Context Is

**Context** is the entire information payload sent to an LLM in a single API call. Since LLMs are **stateless** (no memory across calls), **context** is the only way to give the model everything it needs to respond accurately. It must be reconstructed and sent fresh on every call.

### What Context Includes

**Context** includes:

- **system prompt** -> model role, behavior, constraints
- **few-shot examples** -> optional input-output demos
- **retrieved documents** -> **RAG** chunks from a vector DB
- **conversation history** -> all previous turns
- **current user query** -> the latest request

### Context Assembly Flow

```text
System Prompt
-> Few-Shot Examples (optional)
-> Conversation History
-> Retrieved RAG Chunks (optional)
-> Current User Query
-> Full Context Assembled
-> Tokenization
-> Context Window Check
-> Prefill
-> Decode
-> Response
```

### Why Context Matters

This is one of the most important interview ideas:

- LLMs do not naturally "remember" previous API calls unless that information is passed again
- When people say chatbots have memory, what they usually mean is that the application is rebuilding **context**
- That rebuilt context may include **conversation history**, **summaries**, or **retrieved knowledge**

### Context Window

Every model has a hard token limit called the **context window** (e.g., GPT-4o: 128k, Claude 3.5 Sonnet: 200k, Gemini 1.5 Pro: 1M) that covers both input and output combined.

Important points:

- Exceeding the **context window** causes errors or silent truncation
- Since attention computation is `O(n^2)`, larger contexts mean exponentially higher latency and cost
- A known limitation is the **lost-in-the-middle problem**
- Models pay more attention to the **beginning** and **end** of context
- Critical information in the **middle** is often underweighted

### Context: Why, Where, How, When

- **Why** -> because the model can only respond based on what is present in the current request
- **How** -> by combining system instructions, conversation history, examples, retrieved data, and the latest user query
- **Where** -> in chat applications, assistants, RAG systems, API requests, and multi-turn workflows
- **When** -> whenever the application wants the model to answer accurately using prior conversation or external knowledge

That means **context** is not just a storage problem, it is also an **optimization problem**. You want enough context for accuracy, but not so much that cost, latency, and signal quality get worse. In production, good systems are designed around choosing:

- what to include
- what to summarize
- what to retrieve only when needed

### Context Management Strategies

Context management strategies:

- **Full History** - send all messages every call; simple but token-expensive
- **Sliding Window** - keep only last N messages; cheap but loses early context
- **Summarization** - replace old turns with a compressed summary; balanced
- **Vector Memory (RAG)** - retrieve only relevant chunks per turn from a vector DB; best for long sessions
- **Hybrid** - recent full history + summarized older turns + vector retrieval

If asked which strategy is best, the practical answer is usually **Hybrid**:

- recent turns are kept in full for local coherence
- older turns are summarized for continuity
- external knowledge is retrieved through **RAG** when needed

### KV Cache and Prompt Caching

LLMs internally cache Key-Value attention computations. Providers like OpenAI and Anthropic offer prompt caching discounts when the same system prompt is reused across calls, significantly reducing cost and latency in production.

### Context Poisoning / Prompt Injection

Malicious text in retrieved documents or user input can override system instructions. This is a real security concern in RAG systems where external content is injected into context.

### Production Implications

**Production Implications:**

- Count tokens before sending (use `tiktoken`) to avoid failures
- System prompts consume tokens on every call - keep them tight
- Place critical information at the start or end of context, never the middle
- Sliding window + summarization is the standard pattern for high-traffic chat systems
- Prompt caching is a major cost optimization for repeated large contexts

### RAG (Retrieval-Augmented Generation)

Two concepts that usually come up together with **context** are **RAG** and **hallucinations**.

**RAG** stands for **Retrieval-Augmented Generation**. The system:

1. converts documents into **embeddings**
2. stores them in a **vector database**
3. retrieves the most relevant chunks for a query
4. inserts those chunks into prompt **context**
5. asks the LLM to answer using that grounded information

The main value of **RAG** is:

- it reduces **hallucinations**
- it allows use of **private** or **domain-specific** data
- it avoids putting the entire knowledge base into every prompt

For the storage, indexing, ANN search, metadata filtering, hybrid retrieval, and re-ranking side of this pipeline, see the **Vector Databases** section.

#### RAG: Why, Where, How, When

- **Why** -> to ground the model on relevant external knowledge and reduce hallucinations
- **How** -> retrieve relevant chunks first, then place them into the prompt context before generation
- **Where** -> enterprise search, document Q&A, internal copilots, support bots, research systems
- **When** -> when model pretraining knowledge is not enough or the answer depends on private, recent, or domain-specific data

### Hallucinations

A **hallucination** happens when the model generates incorrect or unsupported information that sounds plausible.

Common reasons include:

- weak **context**
- ambiguous prompts
- missing retrieval
- outdated knowledge
- forcing the model to answer when it should abstain

Good ways to reduce hallucinations are:

- provide clear **context**
- use retrieval
- ask for grounded answers
- require citations when appropriate
- allow the model to say it is uncertain

#### Hallucinations: Why, Where, How, When

- **Why they happen** -> missing context, weak grounding, ambiguous prompts, or overconfident generation
- **How to reduce them** -> better prompts, better retrieval, clearer constraints, verification, and citation requirements
- **Where they appear** -> factual Q&A, summarization, search assistants, enterprise knowledge tools, code generation
- **When they become risky** -> in high-stakes domains such as healthcare, finance, law, research, and internal business workflows

### Prompt Roles

Another related concept is **prompt roles**. In chat-based systems, prompts are often divided into:

- `system`
- `user`
- `assistant`

Meaning of each role:

- `system` -> defines global behavior and rules
- `user` -> carries the latest request
- `assistant` -> represents previous model outputs

From the model's perspective, all of this becomes part of **context**, but the role structure helps shape behavior and instruction priority.

Generation-time controls such as **temperature**, **top_p**, **max_tokens**, and related decoding settings are covered in the **Sampling Parameters** section below.

## RAG - Retrieval Augmented Generation

RAG is an architecture that gives an LLM access to an external, updatable knowledge base at inference time. Instead of relying purely on memorized training weights, the system retrieves relevant information and injects it into the prompt context so the model can answer using grounded, current, and verifiable sources.

RAG directly addresses two major LLM limitations:

- **knowledge cutoff** -> model weights are frozen after training, so they do not include newly created information
- **hallucination** -> when the model does not know something, it may still generate a fluent but incorrect answer

With RAG, the model's role shifts from "recall the answer from weights" to "reason over retrieved evidence and generate the answer from that evidence."

### RAG vs Fine-Tuning

**Fine-tuning** is best for:

- behavior
- style
- response format
- task adaptation

**RAG** is best for:

- factual grounding
- private knowledge bases
- frequently updated data
- source-backed document question answering

Practical production pattern:

- fine-tune for behavior and formatting
- use RAG for factual knowledge

### End-to-End RAG Flow

```text
INDEXING PIPELINE (offline):
Raw Documents
-> Loading + Preprocessing
-> Chunking
-> Embedding Model
-> Dense Vectors + Metadata
-> Vector DB
-> Optional Sparse Index

QUERYING PIPELINE (real-time):
User Query
-> Query Transformation
-> Dense Retrieval + Sparse Retrieval
-> RRF Fusion
-> Cross-Encoder Re-ranking
-> Context Assembly
-> LLM Generation
-> Optional Validation
-> Final Response
```

### Document Loading and Preprocessing

Before chunking and embedding, raw documents have to be loaded and cleaned. This stage is often underestimated, but weak preprocessing silently degrades retrieval quality regardless of how good the later components are.

Common loaders include:

- PDF loaders
- web scrapers
- database connectors
- API connectors
- code and file parsers

Key preprocessing challenges:

- **PDF parsing** -> scanned PDFs need OCR; multi-column layouts can break reading order; tables often get corrupted
- **HTML cleanup** -> navigation, headers, footers, and ads must be stripped
- **deduplication** -> duplicated documents or near-duplicates can dominate retrieval
- **metadata extraction** -> title, page number, author, timestamp, URL, section header, and source information should be preserved

Why this matters:

- better preprocessing improves retrieval quality before any embedding or ANN search happens
- metadata is essential for filters, attribution, and debugging

### Chunking - The Foundation of Retrieval Quality

Chunking is one of the single most important design choices in a RAG pipeline. The embedding model embeds the chunk as a whole. If the chunk is badly formed, retrieval quality drops no matter how good the embedding model or vector database is.

Important chunking approaches:

- **fixed-size chunking** -> split every N tokens with overlap; simple and strong baseline
- **recursive chunking** -> split at paragraph, then sentence, then smaller units if needed
- **sentence-level chunking** -> clean boundaries, but chunk sizes may vary too much
- **semantic chunking** -> split where semantic similarity drops sharply
- **parent-child chunking** -> index small chunks for precision but inject larger parent chunks for richer context

Good default guidance:

- **256-512 tokens**
- **10-20% overlap**

Why chunking matters so much:

- too large -> the embedding mixes unrelated ideas and retrieval becomes noisy
- too small -> the chunk lacks enough context to be useful
- no overlap -> meaning across boundaries gets lost

### Query Transformation - Improving Retrieval Before Search

The user's original query is often not the best retrieval query. Query transformation rewrites or expands the query before retrieval.

Common techniques:

- **query rewriting** -> rewrite the question into a more retrieval-friendly form
- **HyDE** -> generate a hypothetical ideal answer document, embed that, and use it for retrieval
- **multi-query retrieval** -> generate multiple phrasings and retrieve for each
- **query decomposition** -> split a complex question into simpler sub-queries
- **step-back prompting** -> generate a broader, more abstract version of the query and retrieve supporting background context

These techniques are especially useful when there is a vocabulary mismatch between the user's phrasing and the documents' phrasing.

### Retrieval - Dense, Sparse, and Hybrid

**Dense retrieval**:

- uses embeddings
- captures semantics, synonyms, and paraphrases
- may miss exact keywords, IDs, and acronyms

**Sparse retrieval**:

- uses BM25 or related lexical retrieval
- handles exact keyword matching very well
- misses semantic equivalence

**Hybrid retrieval** combines both and is usually the production default. Results from the dense and sparse retrieval paths are often merged with **Reciprocal Rank Fusion (RRF)**.

This is why serious production RAG systems usually avoid dense-only retrieval.

### Re-Ranking - Precision on the Shortlist

ANN retrieval finds the right neighborhood quickly, but ranking inside that neighborhood is often imperfect.

**Bi-encoder retrieval**:

- query and documents are encoded separately
- fast and scalable
- lower precision

**Cross-encoder re-ranking**:

- query and document are processed together
- much better relevance judgment
- much slower, so only used on a small shortlist

Standard production pattern:

```text
Dense + Sparse Retrieval
-> Top-50 Candidates
-> Cross-Encoder Re-ranker
-> Top-5 Chunks
-> Inject into LLM Context
```

Re-ranking is one of the highest-ROI improvements to a naive RAG system.

### Context Assembly

Retrieved chunks must be assembled carefully before being passed to the LLM.

Important considerations:

- **ordering matters** because of the lost-in-the-middle problem
- include **source metadata** for attribution and citations
- respect the **context window budget**
- decide between **stuffing** all chunks at once vs iterative refinement or compression

A practical rule:

- place the most relevant chunks at the beginning and end of the context block

### Advanced RAG Patterns

More advanced RAG systems go beyond a single retrieve-then-generate step.

Important patterns:

- **Self-RAG** -> retrieve, answer, then self-evaluate grounding and faithfulness
- **Corrective RAG (CRAG)** -> add a retrieval-quality evaluator and retry or escalate when retrieval is weak
- **Agentic RAG** -> let an agent decide when and how many times to retrieve
- **Multi-hop RAG** -> perform retrieval over multiple dependent sub-questions
- **Graph RAG** -> retrieve over entities and relationships instead of only chunk vectors

These patterns matter for complex questions where one-shot retrieval is not enough.

### RAG Evaluation

RAG must be evaluated at both the retrieval layer and the generation layer.

**Retrieval metrics**:

- **Recall@K**
- **Precision@K**
- **MRR**
- **NDCG**

**Generation metrics**:

- **faithfulness**
- **answer relevance**
- **context relevance**
- **context recall**

Frameworks like **RAGAs** are commonly used because they separate these concerns more clearly than a single end-to-end accuracy number.

### RAG Failure Modes and Mitigations

Important failure modes:

- **retrieval failure** -> wrong chunks are retrieved
- **context poisoning** -> retrieved documents contain misleading, outdated, or malicious content
- **lost-in-the-middle** -> the model ignores relevant chunks placed in the middle of long context
- **semantic gap** -> user query wording differs too much from source document wording
- **chunk boundary problem** -> the needed answer spans chunk boundaries

Typical mitigations:

- better chunking
- better embeddings
- hybrid retrieval
- re-ranking
- metadata filtering
- query transformation
- context ordering

### Production RAG Architecture

A production RAG system is more than a vector DB plus an LLM call.

Typical production components:

- **document ingestion pipeline** -> handles new, updated, and deleted documents
- **query pipeline** -> transformation, retrieval, re-ranking, context assembly, generation
- **observability** -> log queries, retrieved chunks, scores, context, and responses
- **feedback loop** -> capture user feedback and use it to improve retrieval
- **index management** -> support upserts, deletions, versioning, and re-embedding
- **caching** -> cache embeddings, prompt prefixes, or repeated retrieval results where appropriate

Deletion handling is especially important. Stale chunks from deleted or outdated documents silently degrade retrieval quality and are difficult to debug if the ingestion pipeline does not manage index updates correctly.

### Real-World Usage

- **Notion-style knowledge search** -> workspace documents embedded, filtered, retrieved, and cited
- **GitHub Copilot-style code retrieval** -> code chunks retrieved with code-specific embeddings and parent-child context
- **Enterprise legal AI** -> contracts chunked semantically, re-ranked, cited, and validated before answer delivery
- **Customer support AI** -> combine vector retrieval with CRM or account-specific context for grounded support answers

### Production Implications

- chunking strategy is one of the highest-leverage decisions in RAG
- use hybrid search by default for keyword-sensitive domains
- add a cross-encoder re-ranker early
- place the most relevant chunks at the beginning and end of context
- log full retrieval traces for debugging
- implement deletion handling and index versioning
- evaluate faithfulness and retrieval quality separately
- use query transformation techniques when there is vocabulary mismatch
- for multi-document synthesis tasks, evaluate whether Graph RAG or multi-hop RAG is needed

## Logits

**Logits** are the raw output scores produced by a model before applying any normalization such as **softmax**. They can be **positive**, **negative**, or **zero**, and they are **not probabilities**.

### Why Logits Exist

Why logits exist:

- models compute internal scores for each possible class or token
- these scores show how strongly the model favors each option
- logits are later transformed into probabilities using **softmax**

Example:

```text
Logits: [2.5, 1.0, -0.5]
Softmax: [0.71, 0.21, 0.08]
```

The key idea is:

- higher **logit** usually means higher probability after **softmax**
- the **relative differences** between logits matter more than the absolute values

### Logits: Why, Where, How, When

- **Why** -> because models need a raw scoring stage before converting outputs into probabilities
- **How** -> the final layer produces unnormalized scores, then softmax or sigmoid converts them into usable probabilities
- **Where** -> classification models, transformers, LLM decoding, loss functions such as cross-entropy
- **When** -> whenever a model must choose among classes, labels, or next-token candidates

### Logits in Neural Networks

In neural networks:

- the final layer often outputs **logits**
- then:
  - **softmax** is used for multi-class classification
  - **sigmoid** is used for binary classification

### Logits in Transformers / LLMs

In **Transformers / LLMs**:

- the model outputs a logit for each token in the vocabulary
- **softmax** converts those logits into token probabilities
- the next token is selected either greedily or by sampling

### Important Properties of Logits

Important properties of logits:

- they are **not bounded**
- they can take any real value
- they preserve the ranking of predictions
- many loss functions expect **logits directly**, not probabilities

### Logits vs Probabilities

| Aspect         | Logits       | Probabilities |
| -------------- | ------------ | ------------- |
| Range          | (-inf, +inf) | [0, 1]        |
| Sum            | Not fixed    | = 1           |
| Interpretation | Raw scores   | Likelihood    |

### Common Mistakes

- confusing **logits** with probabilities
- applying **softmax** twice
- ignoring that many loss functions expect logits directly

### One-Liner

Logits are raw, unnormalized scores output by a model that are later converted into probabilities using functions like **softmax**.

### Memory Trick

- **Logits** = scores before probability

## LLM Data Flow (End-to-End)

This section ties together the earlier topics into one full pipeline, from raw input text to generated output token. It is especially useful for interviews because it gives a clean end-to-end explanation of how an LLM actually processes and produces language.

### Step-by-Step Flow

#### 1. Input Text -> Tokenization

- the model receives raw text as input
- a **tokenizer** splits the text into smaller units called **tokens**
- tokens are not always full words

Examples:

- words -> `cat`
- subwords -> `play` + `ing`
- punctuation -> `.` `,`

Example flow:

```text
"The cat sat on the mat"
-> ["The", "cat", "sat", "on", "the", "mat"]
```

Why it matters:

- tokenization helps handle unknown words
- it reduces vocabulary size

#### 2. Tokens -> Token IDs

- each token is mapped to a unique integer using a fixed vocabulary
- this mapping is predefined and consistent

Example:

```text
"The" -> 101
"cat" -> 345
"sat" -> 876

["The", "cat", "sat"] -> [101, 345, 876]
```

These integers are called **token IDs**.

#### 3. Token IDs -> Embeddings

Why this step is needed:

- token IDs are just numbers
- by themselves, they do not carry meaning
- the model needs a meaningful learned representation

An **embedding** is a dense vector representing a token in a high-dimensional space.

How embeddings are created:

- the model contains a large **embedding matrix**
- rows correspond to vocabulary items
- columns correspond to embedding dimensions
- each token ID selects one row from that matrix

Example:

```text
Token ID: 345
-> Embedding: [0.12, -0.45, 0.88, ..., 0.67]
```

Intuition:

- similar words often end up with similar vectors
- very different words end up farther apart

This is one of the key reasons the model can work with meaning instead of just raw symbols.

#### 4. Embeddings -> Transformer Layers

- the embedding vectors are passed through multiple **transformer layers**
- each layer uses **self-attention** to understand relationships between tokens
- each layer also uses **feed-forward networks** to refine the representation

After many layers, the model builds a deep **contextual representation** of the sequence.

#### 5. Final Representation -> LM Head

**LM Head** stands for **Language Model Head**.

- it is the final layer that converts the model's internal representation into scores for every possible token
- it takes the final hidden state from the transformer stack
- it projects that representation to the full vocabulary size

That means the output dimension of the LM head matches the vocabulary size.

#### 6. Output -> Logits

- the LM head produces **logits**
- logits are raw, unnormalized scores
- they may be positive, negative, or zero
- they are not probabilities yet

Example:

```text
mat   -> 2.5
bed   -> 1.2
floor -> 0.5
```

These values show preference, but they are not directly interpretable as probabilities.

#### 7. Logits -> Probabilities

- a **softmax** function converts logits into probabilities

Example:

```text
mat   -> 0.7
bed   -> 0.2
floor -> 0.1
```

Properties:

- values lie between `0` and `1`
- all probabilities sum to `1`

Now the model has a probability distribution over the next token.

#### 8. Selecting the Next Token

The model chooses the next token using a decoding strategy such as:

- **Greedy decoding** -> pick the highest-probability token
- **Sampling** -> sample from the probability distribution

Example:

```text
Chosen token -> "mat"
```

#### 9. Append + Repeat

- the chosen token is appended to the existing sequence
- the updated sequence is fed back into the model
- the whole process repeats to generate the next token

Example:

```text
"The cat sat on the"
-> "The cat sat on the mat"
```

This loop continues until:

- an end condition is reached, or
- a maximum length is reached

### Complete Flow

```text
Raw Text
-> Tokenization
-> Tokens
-> Token IDs
-> Embedding Lookup
-> Transformer Layers
-> LM Head
-> Logits
-> Softmax
-> Probabilities
-> Token Selection
-> Append to Input
-> Repeat
```

### Key Concepts to Remember

- the model generates **one token at a time**
- token IDs are just indices; **embeddings** give them meaning
- the **embedding matrix** is learned during training
- **transformer layers** build contextual understanding
- the **LM head** converts internal understanding into vocabulary scores
- **logits** are raw scores and **softmax** turns them into probabilities
- text generation is an **iterative loop**

### One-Line Summary

Text is tokenized, converted to IDs, mapped to embeddings, processed through transformer layers, converted to logits by the **Language Model Head**, transformed into probabilities with **softmax**, and then the next token is selected and appended repeatedly to generate output.

## Sampling Parameters

**Sampling parameters** are configuration settings passed to an LLM that control how the model selects the **next token** during text generation. They do not change the model's knowledge - they change its **decision-making behavior** during decoding.

At every generation step, the model outputs **logits** over the entire vocabulary. Those logits are converted into probabilities using **softmax**, and then the sampling parameters manipulate either the logits or the resulting probability distribution before the next token is selected.

### Sampling Parameters: Why, Where, How, When

- **Why** -> to control response style, creativity, determinism, repetition, length, and cost
- **How** -> by modifying the token probability distribution before sampling
- **Where** -> chatbots, coding assistants, summarization systems, creative tools, structured generation pipelines
- **When** -> whenever you want to tune model behavior for a specific use case instead of accepting default decoding behavior

The most important idea for interviews is this:

- **Model weights decide what the model knows**
- **Sampling parameters decide how that knowledge is expressed**
- They affect **generation behavior**, not the underlying trained knowledge

### Repetition Penalties

**Repetition penalties** reduce the probability of tokens that have already appeared, helping prevent the model from looping, repeating filler phrases, or getting stuck on the same wording. They are applied directly to **logits before softmax**, which makes them some of the earliest-acting sampling controls in the decoding pipeline.

Pipeline order:

`Logits -> Repetition Penalty -> Divide by Temperature -> Softmax -> Top-K -> Top-P -> Sample`

LLMs naturally tend to repeat because once a token appears in context, attention makes that token more relevant for future predictions. This can create a feedback loop, especially when **temperature** is low and **Top-P / Top-K** are narrow, because the model keeps being pulled toward the same already-high-probability tokens.

**Frequency Penalty** reduces a token's logit in proportion to how many times it has already appeared.

Formula:

`adjusted_logit = original_logit - (frequency_penalty x token_count)`

Key idea:

- a token repeated 5 times is penalized much more than a token seen once
- this is best for reducing habitual filler phrases and word-level repetition in longer outputs
- common API range is `-2.0 to 2.0`

**Presence Penalty** applies a flat one-time penalty to any token that has appeared at least once, no matter how many times it later repeats.

Formula:

`adjusted_logit = original_logit - presence_penalty` if the token has been seen

Key idea:

- this is useful for encouraging new topics and new vocabulary
- it pushes the model away from circling the same concepts
- common API range is `-2.0 to 2.0`

Negative values on either parameter actively encourage repetition, which can be useful in rare cases where repeated patterns are desirable.

**Repetition Penalty** in many open-source or Hugging Face style setups is a unified multiplicative parameter rather than separate additive penalties.

Formula:

`adjusted_logit = original_logit / repetition_penalty` if the token has been seen

Important points:

- `1.0` means no effect
- `1.2` is moderate
- `1.5` is strong
- unlike additive penalties, this scales with the logit magnitude
- strong high-logit tokens get penalized more in absolute terms
- a common starting range for local models is `1.15 - 1.3`

| Parameter    | Frequency Penalty            | Presence Penalty                | Repetition Penalty       |
| ------------ | ---------------------------- | ------------------------------- | ------------------------ |
| Type         | Additive, scales with count  | Additive, one-time              | Multiplicative, one-time |
| Targets      | Habitual word repetition     | Topic stagnation                | General repetition       |
| Best for     | Long-form content, summaries | Creative writing, brainstorming | Local model inference    |
| Typical APIs | OpenAI, Anthropic            | OpenAI, Anthropic               | Hugging Face, Ollama     |

#### Penalty Scope

- some APIs apply penalties to both prompt and completion tokens
- this means terms already present in the user prompt can be penalized in the output
- that can hurt quality when the output should reuse domain-specific terms, variable names, product names, or proper nouns

#### Interaction with Other Parameters

- low temperature + no repetition penalty is a high-risk combination for looping
- high repetition penalties + high temperature can force the model away from its best tokens and reduce coherence
- for code generation, penalties should usually be zero because code naturally repeats keywords, brackets, names, and syntax

#### Real-World Usage

- **Customer Support Bots** -> modest **frequency_penalty** can prevent phrase spiraling
- **Long-form Article Generation** -> a mix of **frequency_penalty** and **presence_penalty** helps keep wording varied across long outputs
- **Local Models (LLaMA / Mistral via Ollama)** -> `repetition_penalty: 1.15 - 1.3` is a common starting point
- **Code Generation** -> set repetition penalties to `0` because repetition is structurally necessary

### Temperature

**Temperature** is a numeric parameter that controls how random or deterministic the model output is. At each generation step, the model produces **logits**, and temperature is applied before **softmax** by dividing each logit by `T`.

Formula:

`P(token_i) = exp(logit_i / T) / sum(exp(logit_j / T))`

This single division reshapes the probability distribution significantly.

```text
Same Logits at different Temperatures:

Tokens:    [cat,   dog,   bird,  fish]
Logits:    [3.0,   2.5,   1.0,   0.5]

T = 0   ->  [1.00,  0.00,  0.00,  0.00]  greedy, always "cat"
T = 0.5 ->  [0.87,  0.12,  0.01,  0.00]  very focused
T = 1.0 ->  [0.60,  0.30,  0.07,  0.03]  default, probabilities as learned
T = 1.5 ->  [0.45,  0.32,  0.14,  0.09]  more spread
T = 2.0 ->  [0.35,  0.28,  0.22,  0.15]  nearly uniform, "fish" now viable
```

How to interpret it:

- At `T = 0`, the model uses **greedy decoding** and always picks the highest-logit token
- At `T < 1`, the distribution becomes sharper, so output is more focused and deterministic
- At `T = 1.0`, probabilities are used roughly as the model learned them
- At `T > 1`, the distribution flattens, so lower-probability tokens become more likely
- Beyond roughly `1.5`, output may become noticeably less coherent

Important clarification:

- **Temperature does not change which tokens are possible**
- It changes **how confidently** the model chooses among possible tokens

#### Temperature and Reproducibility

- `T = 0` is not perfectly deterministic across all providers
- infrastructure differences, floating point behavior, and hardware variation can still create slight changes
- for stronger reproducibility, combine `temperature: 0` with a fixed **seed**

#### Temperature and Top-P Interact

- **Temperature** reshapes the full probability distribution first
- **Top-P** then filters the nucleus from that reshaped distribution
- high temperature + high `top_p` -> maximum randomness
- low temperature + low `top_p` -> maximum focus
- tune only one at a time when possible, leaving the other near default

#### Temperature in Fine-Tuned Models

- fine-tuned models often already have a more peaked output distribution
- even `temperature: 0.7` may produce very focused output if the model is highly confident in that domain
- temperature interacts with the model's learned confidence, not just the architecture

#### Temperature in Multi-Agent Pipelines

- temperature choices compound across chained calls
- a poor high-temperature reasoning step can hurt every downstream step
- common pattern:
  - planning or reasoning agent -> `0.0 - 0.3`
  - generation or writing agent -> `0.7 - 1.0`
  - validator or critic agent -> `0`

#### Temperature by Use Case

| Use Case                  | Temperature | Reason                                   |
| ------------------------- | ----------- | ---------------------------------------- |
| SQL / Code Generation     | 0 - 0.2     | Correctness over creativity              |
| Factual Q&A / Support Bot | 0 - 0.3     | Consistency, lower hallucination risk    |
| Summarization             | 0.3 - 0.5   | Accurate but still natural prose         |
| Conversational AI         | 0.6 - 0.8   | Natural and varied without being robotic |
| Creative Writing / Copy   | 1.0 - 1.2   | More diverse and imaginative output      |
| Brainstorming             | 1.2 - 1.5   | Maximum idea variety                     |

#### Real-World Usage

- **GitHub Copilot / Cursor** -> `T: 0.1 - 0.2`; correctness matters more than creativity
- **ChatGPT-style conversation** -> around `T: 0.7 - 0.8`; natural and varied without becoming too unpredictable
- **Jasper / Copy.ai** -> `T: 1.0 - 1.2` with higher diversity settings for multiple creative outputs
- **Multi-agent pipelines** -> planner `T: 0`, researcher `T: 0.3`, writer `T: 0.8`, critic `T: 0`

#### API Differences

- **OpenAI** -> temperature range commonly `0.0 to 2.0`
- **Anthropic** -> temperature range commonly `0.0 to 1.0`
- always check provider docs because the same numeric value can behave differently across models

### Top-K and Top-P (Nucleus Sampling)

**Top-K** and **Top-P (Nucleus Sampling)** are filters applied after **temperature** reshapes the probability distribution. While temperature changes how confidently the model chooses from the full distribution, **Top-K** and **Top-P** reduce the candidate pool before any token is sampled, cutting away the low-probability tail that often introduces incoherent or nonsensical tokens.

The filtering order is:

`Logits -> Divide by Temperature -> Softmax -> Top-K -> Top-P -> Sample`

**Top-K** keeps only the `K` highest-probability tokens, sets all others to zero, renormalizes the remaining probabilities, and samples from that reduced set.

- `top_k: 50` means only 50 tokens can ever be candidates at that step
- `top_k: 1` is effectively greedy decoding
- the weakness of **Top-K** is that it is a fixed, context-blind cutoff
- if the model is already 95% sure, a large `top_k` still keeps many unnecessary candidates
- if many tokens are equally plausible, a small `top_k` may arbitrarily remove valid options

**Top-P (Nucleus Sampling)** takes an adaptive approach. It sorts tokens by probability and keeps adding them until the cumulative probability reaches `P`, then samples only from that minimal nucleus.

- at `top_p: 0.9`, the candidates are the smallest set of tokens whose total probability reaches 90%
- when the model is highly confident, the nucleus can become extremely small
- when the model is uncertain, the nucleus automatically expands
- this dynamic adjustment is why **Top-P** is generally preferred over **Top-K** in production

```text
TOP-P = 0.9 Adaptive Behavior:

Confident step:
[the: 0.92, cat: 0.05, sat: 0.02, ...]
nucleus = [the] only  <- 1 token, near-greedy

Uncertain step:
[the: 0.12, cat: 0.11, sat: 0.10, dog: 0.10, ...]
nucleus = many tokens <- wide selection
```

When both **Top-K** and **Top-P** are used together, **Top-K** is applied first as a hard ceiling, then **Top-P** selects the nucleus from those remaining tokens. This two-stage approach is common in local-model and some provider pipelines because it reduces the sort space first and then applies adaptive filtering within that smaller set.

#### Repetition vs Coherence Tradeoff

- too wide a candidate pool with high **temperature** + high **top_p** + high **top_k** can introduce incoherence
- too narrow a pool with low **temperature** + low **top_p** + low **top_k** can cause repetitive or looping output
- these parameters should be balanced together instead of treated independently

#### Default Values Across Providers

| Provider                       | Top-P Default              | Top-K Default |
| ------------------------------ | -------------------------- | ------------- |
| OpenAI GPT-style APIs          | 1.0 (effectively disabled) | Not exposed   |
| Anthropic Claude               | 0.999                      | -1 (disabled) |
| Google Gemini                  | 0.95                       | 40            |
| LLaMA / Mistral (local setups) | 0.9                        | 40            |

#### Top-K / Top-P Guidance by Use Case

| Use Case              | Top-P       | Top-K    |
| --------------------- | ----------- | -------- |
| Structured JSON / SQL | 0.3 - 0.5   | 10 - 20  |
| Code Generation       | 0.85 - 0.95 | 40       |
| RAG Document Q&A      | 0.7 - 0.8   | 40       |
| Conversational AI     | 0.9 - 0.95  | 50       |
| Creative Writing      | 0.95 - 1.0  | Disabled |

#### Real-World Usage

- **LLaMA / Mistral (Ollama, LM Studio)** -> `top_k: 40` applied first, then `top_p: 0.9`; common local inference setup
- **Google Gemini** -> often uses `top_p: 0.95` and `top_k: 40` style defaults for coherent professional output
- **Structured JSON extraction** -> tighter `top_p` with low `temperature` reduces malformed output
- **RAG Q&A** -> narrower `top_p` helps keep answers anchored to retrieved content

### Max Tokens

**Max Tokens** is a hard cap on generated output tokens.

- It does not improve quality directly
- It is mainly a **length** and **cost** control
- If the model reaches the limit before finishing, output may stop mid-sentence

Always set **max_tokens** in production to avoid runaway outputs and unexpected billing.

The `finish_reason` tells why the model stopped:

- `end_turn` -> natural completion
- `max_tokens` -> hit the output limit
- `stop_sequence` -> matched a stop sequence

#### Max Tokens: Why, Where, How, When

- **Why** -> to control output length, cost, and latency
- **How** -> by enforcing a hard limit on generated tokens
- **Where** -> all production API calls and agent workflows
- **When** -> always set explicitly in production systems

### Stop Sequences

**Stop Sequences** are strings that immediately halt generation when produced. The stop string itself is excluded from the final output.

They are especially useful for:

- **few-shot prompting**
- **structured generation**
- saving cost by stopping early

Common examples:

- `\n`
- `###`
- `;`

#### Stop Sequences: Why, Where, How, When

- **Why** -> to stop generation exactly where useful output should end
- **How** -> by matching predefined terminating strings during decoding
- **Where** -> few-shot prompts, code generation, SQL generation, role-based prompting, structured text outputs
- **When** -> when you want to prevent the model from continuing into the next example, role, or block

### Seed

**Seed** initializes the random number generator used during sampling, which makes outputs more reproducible.

- Same seed + same inputs + same parameters usually gives the same output
- It is not always 100% guaranteed across infrastructure changes
- It is still very useful for debugging, testing, and A/B comparisons

#### Seed: Why, Where, How, When

- **Why** -> to improve reproducibility
- **How** -> by fixing the random initialization used during token sampling
- **Where** -> testing pipelines, prompt experiments, evaluation workflows, debugging sessions
- **When** -> when you want to compare prompt changes without randomness affecting the result

### Parameter Interaction

**Temperature, Top-K, and Top-P interact** because all three influence token selection. In practice, changing one can amplify or counteract the effect of the others.

Common recommendation:

- adjust only **one** of them at a time
- either keep `top_p: 1` and tune **temperature**
- or keep `temperature: 1` and tune **top_p**
- if **Top-K** is exposed, keep it fixed while tuning one other parameter

### Full Token Generation Flow

```text
Input Prompt
     ->
LLM Forward Pass
     ->
Raw Logits (score for each token)
[0.8, 2.1, 0.3, 5.6, 1.2, ...]
     ->
Apply Repetition Penalty for seen tokens
     ->
Divide by Temperature
     ->
Softmax -> Probability Distribution
[0.001, 0.003, 0.0001, 0.91, 0.002, ...]
     ->
Apply Top-K -> Keep only top K tokens
     ->
Apply Top-P -> Keep nucleus with cumulative probability >= P
     ->
Sample next token from filtered distribution
     ->
Is token in Stop Sequences? -> YES -> Stop generation
                            -> NO  -> Append to output
     ->
Has max_tokens been reached? -> YES -> Stop
                              -> NO  -> Feed back and repeat
     ->
Final Generated Response
```

### Parameter Cheat Sheet by Use Case

| Use Case                  | Temperature | Top-P | Presence Penalty | Max Tokens    |
| ------------------------- | ----------- | ----- | ---------------- | ------------- |
| Factual Q&A / Support Bot | 0 - 0.2     | 1.0   | 0                | Low (100-300) |
| Code Generation           | 0.1 - 0.2   | 0.95  | 0                | Medium        |
| Summarization             | 0.3 - 0.5   | 1.0   | 0.1              | Medium        |
| Creative Writing          | 0.9 - 1.2   | 0.95  | 0.5 - 0.8        | High          |
| Brainstorming             | 1.0 - 1.5   | 0.95  | 0.6              | Medium        |

### Real-World Usage

- **Customer Support Bots** -> `temperature: 0`, `max_tokens: 300`, `stop: ["Human:", "Customer:"]` to prevent role-play continuation in few-shot setups
- **GitHub Copilot / Cursor** -> `temperature: 0.2`, stop sequences at code block delimiters, seed set for reproducibility in tests
- **Creative Tools (Jasper, Copy.ai)** -> `temperature: 1.1`, `presence_penalty: 0.6`, wide `top_p` for maximum variety
- **SQL Generation** -> `temperature: 0`, `stop: [";"]` to stop after the first complete statement

### Production Implications

- Always set **max_tokens** explicitly - never leave it unbounded in production
- Use **finish_reason** to detect truncation and handle it gracefully
- `temperature: 0` does not guarantee identical outputs across all providers because of infrastructure and floating-point differences
- Disable repetition penalties for code, SQL, and strict structured generation when repeated tokens are expected
- Some providers apply repetition penalties to prompt tokens too, so use them carefully when prompt terminology must be preserved
- OpenAI does not expose **top_k**, so use **top_p** only for GPT-style APIs
- When generating structured outputs like JSON, SQL, or XML, tighten **temperature** and **top_p** together
- `top_p: 1.0` effectively disables nucleus sampling
- For local model inference, setting both **top_k** and **top_p** usually gives better control than leaving filtering fully open
- For local open-source models, **repetition_penalty** is often the main defense against degenerate looping
- Do not tune **temperature**, **top_k**, and **top_p** all at once; change one and evaluate
- **Stop sequences** are a cost optimization and are especially useful in structured generation tasks
- **Prompt caching** caches the input prompt, not the output sampling behavior
- For A/B testing model outputs, fix the **seed** to isolate prompt changes from randomness

## AI Agents

An **AI agent** is an LLM that has been given the ability to take actions in the world by using tools. A regular LLM call is passive: you send a prompt, it returns a response, and the interaction ends. An agent is different because it runs in a loop. The LLM acts as the brain, the tools act as the hands, and the system repeatedly reasons, acts, observes, and decides what to do next until the goal is complete.

### Agent Loop

```text
User Goal
-> Agent (LLM as brain)
-> THINK
-> ACT
-> OBSERVE
-> Enough information to answer?
   -> NO  -> loop back to THINK
   -> YES -> Final Response
```

This is the key conceptual difference between a normal LLM call and an agent:

- a normal LLM call answers once
- an agent can operate across many steps
- tool results are fed back into context
- the model keeps deciding what to do next until it completes the task

### The ReAct Pattern - Reasoning + Acting

The dominant pattern for agent systems is **ReAct (Reasoning + Acting)**.

At each loop iteration, the model conceptually produces:

- **Thought** -> what it knows, what it lacks, and what it should do next
- **Action** -> the tool call it wants to make
- **Observation** -> the tool result that comes back and is injected into context

Why it matters:

- explicit reasoning improves tool choice
- acting connects the LLM to external systems
- observation allows the model to revise its strategy step by step

**Example - Travel Planning Agent:** a user asks for the cheapest flight under a budget. The agent searches flights, checks payment information, books the option that fits the goal, and only then returns the final confirmation. The important point is that no single prompt contains the whole answer in advance; the answer is built through the loop.

### Tool Use - How Agents Interact With the World

Tools are functions the agent can invoke. The LLM does not execute code directly. Instead, it emits a structured tool call such as a function name plus JSON arguments. The runtime executes the tool, returns the result, and injects that result back into the model context.

This is how the model's reasoning is connected to real-world actions.

Common tool categories:

- **information retrieval** -> web search, vector DB lookup, SQL queries, document readers
- **code execution** -> Python, JavaScript, shell, notebooks, sandboxes
- **external APIs** -> weather, payments, CRMs, maps, booking systems
- **file operations** -> reading, writing, parsing PDFs, spreadsheets, documents
- **browser control** -> click, fill, navigate, scrape
- **communication** -> email, Slack, calendar, ticketing
- **memory operations** -> read from or write to memory stores

Tool descriptions matter a lot. The model selects tools from their descriptions, so vague or generic tool descriptions produce weak tool selection. Good descriptions explain:

- when to use the tool
- when not to use the tool
- what inputs it expects
- what kind of result it returns

### Memory - How Agents Remember

A pure LLM is stateless. Agents need memory in order to handle long tasks and, in many cases, multi-session behavior.

Four important memory types appear in agent systems:

- **In-context memory** -> current working memory inside the context window
- **External memory** -> vector-store memory of past interactions and prior tasks
- **Entity memory** -> exact structured facts such as user profile, account details, preferences, project state
- **Procedural memory** -> knowledge and skills stored in model weights from training

**Example - Personal Assistant Agent:** the agent may retrieve the user's earlier meeting preferences from vector memory, look up the user's calendar account from entity memory, reason in-context about the current request, and then use a calendar API to act. All these memory layers work together.

### Planning - How Agents Break Down Complex Goals

Simple ReAct is often enough for straightforward tasks. More complex tasks need explicit planning.

Important planning styles:

- **Plan-and-Execute** -> create a plan first, then execute each step
- **Tree of Thoughts (ToT)** -> explore multiple possible reasoning branches
- **MCTS (Monte Carlo Tree Search)** -> search over possible trajectories and estimate which are most promising

Why planning matters:

- improves reliability on long tasks
- reduces random wandering
- makes multi-step goals more manageable

**Example - Market Research Agent:** instead of immediately searching randomly, the agent first creates a plan such as finding competitors, collecting pricing, gathering reviews, and then synthesizing the results into a final comparison.

### Multi-Agent Systems

Single agents eventually hit limits:

- context windows fill up
- one agent may not be specialized enough
- there is no parallelism

Multi-agent systems solve this by using multiple specialized agents coordinated by an orchestrator.

```text
User Goal
-> Orchestrator Agent
-> Subtask Decomposition
   -> Research Agent
   -> Code Agent
   -> Data Agent
   -> Writer Agent
-> Orchestrator Aggregates Results
-> Critic / Validator Agent
-> Final Response
```

Common communication patterns:

- **Sequential** -> one agent's output becomes the next agent's input
- **Parallel** -> multiple agents work independently at the same time
- **Hierarchical** -> orchestrators delegate to sub-orchestrators and workers
- **Debate / Critique** -> multiple agents produce candidates and another agent critiques or selects

Common frameworks:

- **LangGraph** -> graph-based workflows with conditional edges
- **AutoGen** -> agents as conversational entities
- **CrewAI** -> role-based agent crews
- **OpenAI Assistants / managed agent runtimes** -> hosted tool-calling and workflow infrastructure

### Agent State Management Across Long Tasks

As agents run for many steps, context can fill up with:

- tool results
- observations
- intermediate reasoning
- plan progress

State-management strategies include:

- **summarization** -> compress old history into compact summaries
- **working-memory pruning** -> drop raw observations once their conclusions are retained
- **external state stores** -> persist intermediate outputs outside the prompt
- **checkpointing** -> save task state so long-running jobs can resume after interruption

These are not optional at scale. Without them, long-running agents will eventually exceed context limits or become expensive and unstable.

### Agentic Failure Modes and Reliability

Agents are less reliable than single-turn LLM calls because mistakes compound across steps.

Important failure modes:

- **hallucinated tool calls** -> wrong filenames, wrong endpoints, wrong table names, bad parameters
- **infinite loops** -> the agent repeats the same failed action again and again
- **reward hacking / goal misalignment** -> the agent technically satisfies the goal but not the user's true intent
- **context poisoning / prompt injection** -> malicious tool outputs manipulate the agent
- **irreversible actions** -> sending emails, deleting files, making purchases, or changing production systems unsafely

Typical mitigations:

- strict tool schemas and validation
- step limits
- loop detection
- sanitizing tool outputs
- explicit permissions
- human checkpoints for irreversible actions

### Human-in-the-Loop (HITL)

Full autonomy is fine only for low-stakes and reversible tasks. High-stakes or irreversible actions require **human-in-the-loop** controls.

Common HITL patterns:

- **interrupt and confirm** -> pause before an irreversible action
- **approval workflows** -> present plan before execution
- **async review** -> perform work in draft form and wait for approval before finalizing

This is one of the biggest differences between a demo agent and a production agent.

### Agent Evaluation

Evaluating agents is harder than evaluating a single LLM answer because the final output is only one part of the system.

Important evaluation dimensions:

- **trajectory evaluation** -> assess the full sequence of thought, action, and observation
- **task completion rate** -> percentage of tasks completed successfully end-to-end
- **step efficiency** -> how many steps were used relative to what was necessary
- **tool call accuracy** -> whether the right tools and parameters were chosen

A correct final answer can still come from a brittle or unsafe trajectory, so the full path matters.

### Real-World Usage

- **GitHub Copilot Workspace** -> orchestrates code understanding, implementation, testing, and PR generation across multiple subtasks
- **Devin-style engineering agents** -> combine terminal use, code execution, web browsing, and debugging loops
- **Customer Support Agents** -> combine retrieval, CRM lookups, ticketing tools, and escalation logic
- **LangGraph / AutoGPT-style systems** -> graph-based or loop-based multi-step task execution with tool use

### Production Implications

- always set a maximum step limit
- require HITL for irreversible actions such as payments, deletions, or outbound communication
- invest heavily in precise tool descriptions
- log the full agent trajectory for debugging
- implement context-window management for long-running tasks
- sanitize tool outputs before injecting them into the context
- start with single-agent ReAct before introducing multi-agent complexity
- evaluate trajectory quality, not just final answers
