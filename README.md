# Effects of Quantization on Neural Network Learning

**MSc Dissertation — AI & Adaptive Systems, University of Sussex, 2021**

> Do Bayesian Neural Networks handle quantization better than standard networks? And what does that mean for running AI on edge devices without catastrophic forgetting?

---

## The problem

Billions of IoT devices and smartphones generate data continuously — but most ML inference happens in the cloud, introducing latency, bandwidth costs, and privacy risks. **Edge AI** solves this by running models on the device itself, but edge hardware is constrained: limited memory, limited compute.

The solution is model compression. The most common technique: **quantization** — reducing weights from 32-bit floats to 8-bit integers, achieving up to 4x size reduction and 2-4x faster inference.

The catch: quantization introduces precision loss. And in **continual learning** (where a model must learn new tasks without forgetting old ones), that precision loss can interact badly with catastrophic forgetting.

This dissertation asks: **are Bayesian Neural Networks better suited for Quantization-Aware Training than standard networks — especially in the context of continual learning?**

---

## The answer: yes, significantly

| Experiment | Standard MLP accuracy drop (QAT) | Bayesian MLP accuracy drop (QAT) |
|---|---|---|
| Fashion MNIST (single task) | −3.79% | **−0.01%** |
| SVHN (single task) | +12.54% (improved) | **−0.04%** |
| 5-Split MNIST (continual) | −1.53% | **0.00%** |
| Permuted MNIST (continual) | −18.86% | **−1.48%** |
| 5 mixed datasets (continual) | −3.06% | **−0.07%** |

Bayesian Neural Networks (BNNs) trained with Uncertainty-guided Continual Learning (UCB) showed near-zero accuracy degradation under quantization across every benchmark — while standard MLPs degraded significantly, especially on harder continual learning tasks.

**Key finding on continual learning:** quantization preserves the number of parameters (just at lower precision), whereas pruning removes them entirely. The experiments suggest that **parameter count matters more than precision** for continual learning — a quantized model retains the capacity to remember previous tasks even in compressed form.

---

## Approach

### Two models compared

| | Standard MLP | Bayesian MLP (UCB) |
|---|---|---|
| Weights | Point estimates | Probability distributions |
| Uncertainty | None | Explicit (σ per parameter) |
| Continual learning | Naive SGD | Learning rate ∝ 1/uncertainty |
| Compression | QAT to int8 | QAT applied (size not reduced — PyTorch limitation at time of writing) |

### Uncertainty-guided Continual Learning (UCB)

Parameters with **high uncertainty** (wide distribution) are available to learn new tasks — larger learning steps allowed. Parameters with **low uncertainty** (narrow distribution, already learned) are protected — smaller learning steps, preventing forgetting.

```
Importance Ω ∝ 1/σ

Learning rate for σ  ←  α_σ / Ω_σ
Learning rate for ρ  ←  α_ρ / Ω_ρ
```

This means the network self-regulates: it knows which weights it's confident about and protects them automatically — no external memory or task labels needed.

### Quantization-Aware Training (QAT)

Rather than quantizing after training (post-training quantization), QAT simulates int8 precision *during* training using fake quantization:

```
x_out = FakeQuant(x) = s · (Clamp(round(x_in / s) − z) + z)
```

The model learns to be robust to the precision loss before it's actually applied — resulting in much smaller accuracy drops than post-training quantization.

---

## Benchmarks

| Dataset | Type | Tasks |
|---|---|---|
| Fashion MNIST | Single-task | 10 clothing classes |
| SVHN | Single-task | Street View digit recognition |
| 5-Split MNIST | Continual (class-incremental) | 5 pairs of digits, sequential |
| Permuted MNIST | Continual | 10 random pixel permutations |
| MNIST + SVHN + FashionMNIST + CIFAR10 + notMNIST | Continual (multimodal) | 5 different datasets sequentially |

---

## Why this matters now

This research was published in 2021. Since then, every major trend has made it more relevant:

**On-device AI is mainstream** — Apple Neural Engine, Google Tensor, Qualcomm NPUs all run quantized models. Understanding *how* quantization interacts with learning is no longer academic.

**Continual learning is unsolved at scale** — LLMs are still largely retrained from scratch. The catastrophic forgetting problem this dissertation addresses is one of the core open problems in making AI systems that genuinely learn over time.

**Bayesian uncertainty = better AI products** — the uncertainty quantification in BNNs maps directly onto the PM challenge of building AI features users can trust. A model that knows what it doesn't know is a better product, not just a better model.

**Edge AI + privacy** — with increasing regulatory pressure on data residency (GDPR, AI Act), on-device inference is becoming a product requirement, not just a performance optimisation.

---

## Technical stack

- **Language:** Python
- **Framework:** PyTorch (QAT module, fbgemm backend for x86)
- **Architecture:** Multi-head MLP (shared layers + task-specific heads)
- **Baseline:** UCB implementation by Ebrahimi et al. (ICLR 2020)

---

## Read the dissertation

Full paper: https://github.com/marci6/Effects-of-Quantization-on-Neural-Network-Learning/blob/main/Effects%20of%20Quantization%20on%20Neural%20Network%20Learning.pdf

---

## Context

MSc dissertation completed at the **University of Sussex, 2021**, supervised by Dr Novi Quadrianto. It demonstrates:

- Bayesian deep learning and variational inference
- Continual / lifelong learning methods
- Model compression for edge deployment
- Rigorous experimental design across 5 benchmarks
- Ability to situate applied ML work within broader research context

---

*Marcello Chiesa — [LinkedIn](https://linkedin.com/in/marcellochies) · [GitHub](https://github.com/marci6)*
