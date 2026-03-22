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

### Training Pipeline

```text
Raw Text Corpus -> Tokenization -> Pretraining (next-token prediction)
-> Pretrained Base Model -> SFT -> RLHF / DPO / RLAIF
-> Aligned Assistant Model
```

**Pretraining** usually uses **next-token prediction**. The model sees a token sequence and learns to predict the next token. This objective is simple, self-supervised, and extremely scalable. At large enough scale, it forces the model to internalize grammar, factual associations, coding patterns, and many reasoning-like behaviors.

**Post-training alignment** then turns a raw base model into a usable assistant:

1. **SFT (Supervised Fine-Tuning)** -> train on instruction-response examples
2. **RLHF** -> use human preference rankings and optimize the model toward preferred outputs
3. **RLAIF / Constitutional AI** -> use AI feedback guided by principles instead of relying only on human labeling
4. **DPO (Direct Preference Optimization)** -> a simpler preference-optimization alternative that avoids the full RLHF loop

### Inference

LLMs generate output **autoregressively**, one token at a time. Each newly generated token is appended back into the context and used to predict the next one.

Two phases are useful to remember:

- **Prefill** -> the input prompt is processed in parallel
- **Decode** -> output tokens are generated sequentially

This is why long prompts and long outputs behave differently in latency. During decode, **KV Cache** stores attention key-value states so earlier tokens do not need to be recomputed every step. This is one of the most important inference optimizations in production. Streaming responses show tokens as they are produced during decode.

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
