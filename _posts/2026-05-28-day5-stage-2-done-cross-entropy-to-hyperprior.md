---
title: "Day 5 — Stage 2 done: the four steps from CrossEntropyLoss to my advisor's paper"
date: 2026-05-28 21:50:00 +0200
categories: [Learned-compression]
tags: [cross-entropy, KL-divergence, autoencoder, quantization, hyperprior, rate-distortion, information-theory]
math: false
---

> Day 5 was the day my advisor's ICIP 2025 paper stopped being a wall of unfamiliar symbols and became a sentence I could read. The walk took four steps: **cross-entropy and KL divergence** (lesson 7b — decoding the loss I'd been using since Day 2 without fully understanding it), **autoencoders** (lesson 8 — the structural skeleton of every learned-compression paper I'll read for the next six months), **quantization** (lesson 9 — the unsexy step that turns out to be the bridge from differentiable neural networks to bits-on-disk), and **hyperprior** (lesson 10 — the trick that took learned compression from "okay" to "state of the art" around 2018, and that my advisor's paper is built on top of).
>
> By the end of the day the loss function in her paper — the one I'd been staring at for weeks — read line by line as a thing I could explain to someone else. That is the graduation line for stage 2. From here, the road is reading actual literature and starting experiments in parallel with thesis discussions, not learning from a curriculum.

## 1. Cross-entropy is what I'd been using all along

I used `nn.CrossEntropyLoss()` to train MNIST on Day 4 without really understanding what it measured. Lesson 7b filled in the gap by tying it back to the entropy story from Day 4.

The setup is this. For one MNIST image whose true label is "7", the **true distribution** is a one-hot vector `p = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]` — all the probability sits on class 7, none on the others. The **network's prediction** is a softmax distribution `q` — some probability on each of the 10 classes. The cross-entropy between them is:

$$H(p, q) = -\sum_{i=0}^{9} p(i) \log q(i)$$

Because `p` is one-hot, every term in the sum is zero except the one for the true class. So this collapses to **`loss = −log q(true class)`**, which is exactly what `nn.CrossEntropyLoss()` computes (modulo natural-log vs log₂; PyTorch uses natural log, so the units are nats, not bits, but the shape is identical).

This unlocks the Day 1 mnemonic. **Confidently right < hesitantly right < hesitantly wrong < confidently wrong** isn't a slogan; it's the literal shape of `−log` near 0 and 1. A confidently-right prediction (`q(true)` near 1) makes `−log q(true)` go to zero. A confidently-wrong prediction (`q(true)` near 0) makes `−log q(true)` blow up. Cross-entropy *is* that asymmetry.

**KL divergence** then completes the picture. It is the gap between using your imperfect `q` as a codebook and using the optimal `p` codebook:

$$D_{KL}(p \,\|\, q) = H(p, q) - H(p) \geq 0$$

For classification, `H(p) = 0` (the one-hot truth has no uncertainty), so cross-entropy literally *is* the KL divergence in this case. Minimising cross-entropy is minimising KL, which is making `q` match `p`. The ideal endpoint is `q = p`, at which point the loss is zero. That is the *theoretical* training target. (No real model gets all the way there.)

## 2. Autoencoder — the structural skeleton

Lesson 8 introduced the architecture that every learned compression paper is built on. It has the shape of an hourglass:

- An **encoder** maps the input `x` (say, an MNIST image's 784 pixels) down through a series of layers to a **bottleneck** `z` (a vector of much lower dimension, e.g. 8 floats).
- A **decoder** maps `z` back up to a reconstruction `x'` of the original size.
- They're trained together by minimising the reconstruction loss `||x - x'||²`.

The bottleneck is the load-bearing piece. All the information has to squeeze through `z` before reaching the decoder, so the encoder is forced to learn what minimal facts about `x` are sufficient for the decoder to reconstruct it.

What surprised me on this lesson was understanding *why* the autoencoder can compress MNIST 784 → 8 → 784 without complete loss. The trick isn't magic; it's that MNIST data, despite living in a 784-dimensional ambient space, actually occupies a *much* lower-dimensional **manifold** inside that space. There are only ten digits, and within each digit there's a small number of meaningful degrees of variation (slant, thickness, size, style). The "true" intrinsic dimensionality of MNIST is probably closer to 10–20 than to 784, and the bottleneck z just needs to be wide enough to parameterise that manifold.

A pure random-noise image cannot be autoencoded down to 8 floats — its 784 dimensions are genuinely independent, the manifold is the entire space, and the bottleneck would have to be 784-wide. Autoencoding works on real data *because* real data has structure beneath the surface, and the autoencoder discovers a coordinate system for that structure. **Learned compression as a whole rests on this fact.**

The bottleneck dimension is also the first appearance of the rate–distortion trade-off. A small bottleneck (z = 2) is high compression but blurry reconstruction; a large bottleneck (z = 256) is excellent reconstruction but barely any compression. The interesting design choices live somewhere in between.

## 3. Quantization — the unsexy step that matters

The encoder produces `z` as a vector of *floats*. A bitstream stores *bits*. You can't write a float onto a bitstream losslessly without either spending 32 bits per number (wasteful, ignores the actual distribution) or first converting the float into a discrete integer that can be entropy-coded efficiently. **That conversion is quantisation.**

The simplest version is just rounding:

```python
z_q = z.round()    # 3.14 → 3,  2.71 → 3,  -1.41 → -1,  ...
```

For all its simplicity, two non-trivial details fall out of this:

**Quantisation is not differentiable.** The `round` function is a staircase: zero derivative almost everywhere, infinite (Dirac) at integer boundaries. If you put a literal `round` in the middle of your encoder–decoder pipeline, the gradient signal from the reconstruction loss can't flow back through it to the encoder. The entire pipeline is broken for training.

There are two standard fixes. The **straight-through estimator** (STE) pretends `round`'s gradient is 1 in the backward pass (the identity function); rough, but workable. The **noise replacement trick** (Ballé 2018) replaces `round(z)` with `z + uniform(−0.5, 0.5)` during training — a smooth, differentiable surrogate — and switches back to actual rounding at test time. The noise trick is what nearly all modern learned-compression papers use, including (presumably) my advisor's.

**The quantisation step size is another rate–distortion knob.** Finer steps → more precise `z_q` → higher entropy → more bits to encode → less distortion. Coarser steps → less precise `z_q` → lower entropy → fewer bits → more distortion. Setting the step is the same kind of decision as setting the bottleneck width, just expressed in a different parameter.

I had been carrying around "quantisation" as a single word for "make it discrete", without seeing that it carries two distinct technical problems (the differentiability issue and the rate–distortion knob), with two well-developed solutions. Both solutions are now visible in any compression paper I open.

## 4. Hyperprior — turning marginal entropy into conditional entropy

Lesson 10 was the headline lesson of the day. Hyperprior is the technique that took learned image compression from "competitive with JPEG" to "beating BPG and rivalling H.265" around 2018, and it is the direct technical precursor to my advisor's paper.

The naive approach to entropy-coding `z_q` is to assume some fixed probability distribution for it (e.g., zero-mean Gaussian with unit variance) and let the arithmetic coder use that. This works, but is suboptimal: different regions of an image produce `z_q` values with very different distributions (a smooth region produces near-zero values; a textured region produces a wider spread), and a single fixed model fits none of them well. The cost of this mismatch shows up as extra bits in the bitstream.

**Hyperprior is a second neural network that produces "side information" `h` describing the local distribution of `z`.** The full architecture has two routes:

```
   x ──Encoder──► z ──Quantise──► z_q
                  │
                  └─HyperEncoder──► h ──Quantise──► h_q
                                                    │
                                                    └─HyperDecoder──► (μ, σ) for z_q
```

Both `h_q` and `z_q` go into the bitstream. The decoder reads `h_q` first, uses it to predict the distribution `p(z_q | h_q)` — typically a Gaussian whose mean and variance are predicted from `h_q` — and then arithmetic-decodes `z_q` against that conditional distribution. The bottom line is that the rate becomes:

$$R \approx H(h_q) + H(z_q \mid h_q)$$

The trade is that you pay a small extra rate to transmit `h_q`, in exchange for a *large* reduction in the rate for `z_q`. Why does this come out positive? Because of the most useful inequality in this whole story:

$$H(z_q \mid h_q) \leq H(z_q)$$

**"Conditioning never increases entropy"** — knowing a hint about `z_q` always reduces the bits needed to send it, never increases. Pay a tiny rate for a precise hint, save a much larger rate on the main payload, end up with a smaller bitstream overall. That is the *engineering* point of the whole hyperprior architecture.

This was the moment in the day where the conceptual chain from Shannon (1948) to the advisor's paper (2025) became one continuous thread.

## 5. Reading my advisor's loss function

By the end of lesson 10 I could open my advisor's paper and read the joint loss function line by line:

$$\mathcal{L} = \alpha \cdot \text{rate} + \beta \cdot \text{distortion} + \gamma \cdot \text{task}$$

| Term | What it actually is |
|---|---|
| `rate` | `H(h_q) + H(z_q ∣ h_q)` — the marginal entropy of the hyperprior side info plus the conditional entropy of the main latent given the side info |
| `distortion` | `‖x − x'‖²` — the reconstruction MSE we have seen all week |
| `task` | A downstream task-specific loss — cross-entropy for classification, mAP-shaped losses for detection, etc. — applied to whatever predictions a downstream model makes from the reconstruction |

`α`, `β`, `γ` are scalar weights that trade off the three objectives. The novel contribution of the paper is the `γ · task` term: most prior learned-compression papers care only about rate and distortion (compress small, reconstruct well); my advisor's paper additionally requires the reconstruction to be *useful for a downstream machine task*. That is precisely the slice of the field she has staked out for her group, and the thesis area she has offered me.

I could not read this paper four days ago. I can read it now. That is the entire point of the four-day curriculum I have just walked through, and it is precisely the place I want to be when I sit down with her to actually discuss thesis topics.

## 6. What stage 2 unlocked

Stage 2 was always going to be the bridge: between the foundations of deep learning (Stage 1, finished on Day 4) and the actual literature of the thesis area (Stage 3 and beyond). With Stage 2 complete, two things change:

- **I can now read papers in the area without papering over gaps.** The 28-paper literature scan I did before Day 1 was painful precisely because terms like "hyperprior", "rate-distortion loss", and "entropy bottleneck" were treated as nouns to be respected rather than concepts to be understood. The plan from here is to go back to that scan and re-read the five or six most important papers — Bits-to-Photon, HAC++, SA-3DGS, CSGaussian, and PCGCv2 — now that I can actually parse what they're doing.
- **Learning and doing can now run in parallel.** Stage 3 (point clouds, sparse conv, 3DGS, scalable compression, task-driven compression) is best learned alongside actually running experiments. The "learn for four days, then start working" mode is finished; from here it is "read a paper, write a small reproduction, refine my reading of the paper". This is the regime I am going to spend the next six months in.

## Stuck points

By blog convention every post closes with the day's stuck points — what I got wrong and how I corrected it.

1. **KL divergence ≥ 0 is a statement about a single number, not a pointwise inequality.** I had been carrying a confused picture where "`D_KL(p∥q) ≥ 0`" meant something like "`q(x) ≥ p(x)` everywhere". It does not. `q(x)` and `p(x)` can cross each other freely across `x`; what the inequality says is that the *single scalar* that measures the overall difference between the distributions is always non-negative. That scalar reaches zero only when the entire `q` distribution matches the entire `p` distribution. The "distance-between-two-cities is non-negative" metaphor finally pinned this down for me.
2. **Autoencoders compress data because data has low-dimensional structure, not because the network is doing magic.** I had been mentally treating the bottleneck as a kind of clever trick that "summarises" the input. The far better picture is: real-world data sits on a low-dimensional manifold inside its high-dimensional ambient space; the autoencoder discovers a coordinate system for that manifold. Random noise is the limiting case where the manifold is the entire space — and it cannot be autoencoded. The dependency on data structure is essential, not incidental.
3. **Quantisation breaks training, and the trick to repair it is *the* universal pattern of "differentiable proxy".** The `round` function's zero-derivative-almost-everywhere is a real problem and not a corner case. The straight-through estimator and the additive-noise trick are not workarounds; they are the standard pattern that crops up *everywhere* in modern ML wherever you need to put a non-differentiable operation inside a differentiable pipeline. Recognising this as a pattern (not as a one-off compression trick) is itself the lesson.
4. **Hyperprior is not a different kind of model — it is a way to convert marginal entropy into conditional entropy.** I had been treating "hyperprior" as a piece of jargon attached to a specific 2018 paper. Once I saw the `H(z|h) ≤ H(z)` inequality and the "pay a tiny rate for `h`, save a large rate on `z`" trade, the conceptual content of the whole architecture became one sentence: *condition the entropy of the main payload on a small piece of learned side information*. Every learned-compression paper since 2018 that gets a state-of-the-art result is doing some refinement of this single move.

*This is Day 5 — Stage 2 done. From Day 6 onward, the posts will shift in shape: from "I learned X" to "I read paper Y and here is what it actually says", and eventually to "I ran experiment Z and here are the results". The thesis-work mode begins.*
