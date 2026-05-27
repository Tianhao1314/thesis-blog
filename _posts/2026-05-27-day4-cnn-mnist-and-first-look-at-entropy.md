---
title: "Day 4 — A working CNN, where the parameters really live, and the first look at entropy"
date: 2026-05-27 17:40:00 +0200
categories: [Foundations]
tags: [PyTorch, CNN, convolution, MNIST, information-theory, entropy, parameter-counting]
math: false
---

> Day 4 was the day I crossed the line where everything we had been talking about as abstraction became a thing I could actually run. A small convolutional network, written from scratch, trained for under five minutes on a Mac CPU, got 98.68% test accuracy on MNIST. The line between "I have read about deep learning" and "I have done deep learning" sat exactly there.
>
> This post is the log of that line. It also picks up two unexpected lessons along the way — about *where the parameters of a CNN actually live*, and about *what entropy is when you stop using the word "uncertainty" and start using "the lower bound on how many bits you need to send"*. Both are the bridge that takes me from the foundations into the thesis area (learned compression of point clouds and 3D Gaussian Splatting).

## 1. Why MLPs cannot do images

Until today my mental picture of a neural network was the kind of network we had been training in lessons 1–5: stack a few `nn.Linear(...)` layers, put ReLUs between them, train. That structure has a name — **MLP**, *multi-layer perceptron* — and it works fine for tiny structured tasks like the 1D linear regression we hand-fit yesterday.

But the moment you point an MLP at an image, three problems show up at once.

**Parameter explosion.** A 256×256 grayscale image has 65,536 pixels. A first `Linear(65536, 1000)` layer is **65,536 × 1,000 + 1,000 ≈ 65 million** parameters in *one* layer. Even ignoring training cost, the data needed to specify 65 million parameters reliably is enormous. The first layer of an MLP on images is already past the point of viability for ordinary research setups.

**No position invariance.** Worse than the count: each of those 65 million weights *belongs to a specific pixel*. The weight connecting pixel (10, 30) to neuron #1 of the first hidden layer is a *different* weight than the one connecting pixel (180, 200) to neuron #1. If the MLP learns "a cat ear in the top-left corner looks like (10, 30) dark, (12, 32) bright", it has *not* learned to recognise a cat ear in the bottom right — that would require a second, completely independent set of weights covering (180, 200) and (182, 202). For every position. The MLP has to re-learn the same feature at every possible location.

**No spatial awareness.** The MLP takes the image flattened into a 1D vector of 65,536 numbers. As far as it is concerned, pixel (10, 30) and pixel (10, 31) — which are *neighbours* in the original 2D image — are just "the 2570-th and 2571-st numbers in a long list", with nothing telling the network that they are adjacent. Real image regularities all rest on the fact that nearby pixels are correlated; an MLP has to discover that fact from data, the hard way, every time.

CNN is the answer to all three of these problems with one idea: instead of giving every neuron full access to every pixel, **give every neuron a tiny window that only sees a 3×3 patch — and then slide the same window across the entire image**.

## 2. What a CNN actually does

A **convolution layer** maintains a set of small **filters**, each one a 3×3 grid of weights (in my TinyCNN, `Conv2d(1, 16, kernel_size=3)` means 16 filters, each 3×3, applied to a 1-channel input). For each filter, the operation is:

- Lay the filter on top of a 3×3 patch of pixels.
- Elementwise multiply, sum the result. That single number is the response of the filter at that location.
- Slide the filter to the next position. Repeat across the entire image.
- The output is a 2D **feature map** — a grid of responses showing where that particular feature lit up.

Two structural consequences follow from this design:

**Weight sharing.** The 3×3 filter is the *same* at every position. If the filter happens to be tuned to detect a vertical edge, then a vertical edge anywhere in the image — top-left, dead centre, bottom-right — produces the same response. The network never has to re-learn that feature at a new position.

**Tiny parameter count.** A 3×3 filter has 9 weights and 1 bias. **Ten parameters total**, regardless of how large the input image is. Sixteen filters in one conv layer: 160 parameters. Compare to the 65-million-parameter first layer of an MLP on the same input.

The complete CNN architecture I used:

```
Input: 1 × 28 × 28 (grayscale MNIST digit)
  ↓ Conv(1→16, 3×3) + ReLU      → 16 × 28 × 28
  ↓ MaxPool(2×2)                 → 16 × 14 × 14   (spatial halved)
  ↓ Conv(16→32, 3×3) + ReLU      → 32 × 14 × 14
  ↓ MaxPool(2×2)                 → 32 × 7 × 7     (spatial halved again)
  ↓ Flatten                       → 1568
  ↓ Linear(1568 → 64) + ReLU      → 64
  ↓ Linear(64 → 10)               → 10  (one number per digit class)
```

Five distinct operations carry the whole thing:

- **Conv** extracts local features through filters.
- **ReLU** zeros out the negative part so non-linearities can stack.
- **MaxPool** halves the spatial size by keeping the largest value in each 2×2 region — preserving "where the feature was strong" while discarding precise pixel location.
- **Flatten** unrolls the final feature stack into a 1D vector for the classifier.
- **Linear** is the final classification head: it has access to the entire flattened feature vector and outputs one score per class.

The architecture's pattern is unmistakable once you see it: **spatial size shrinks (28 → 14 → 7), channel count grows (1 → 16 → 32)**. Early layers cover the whole image at fine resolution looking for simple features (edges, blobs); deep layers cover the same image at coarse resolution, looking for complex features composed out of the early ones.

## 3. The training run, and what 98.68% means

The training loop is the same one we wrote yesterday in Day 2, with two differences: the optimiser is Adam (lesson learned from Day 3), and the loss is `CrossEntropyLoss` (because the task is classification, not regression). 3 epochs over the 60,000-image MNIST training set. About four minutes on my Mac's CPU.

```
Epoch 1: avg train loss = 0.3358   test accuracy = 97.46%
Epoch 2: avg train loss = 0.0790   test accuracy = 98.28%
Epoch 3: avg train loss = 0.0566   test accuracy = 98.68%
```

10 randomly-sampled test images: all 10 predicted correctly.

98.68% on 10,000 images means **9,868 correct, 132 wrong** — out of digits the network had *never seen during training*. The number itself is not surprising; MNIST is the famously-easy benchmark and 98–99% is standard for any honest CNN. What is surprising — to me, having walked the whole road by hand — is how *little* code it took to get there. Roughly 50 lines, half of which is comments. From "what is deep learning" on Day 1 to a working CNN on real data four days later, the path is tighter than I would have guessed.

The training-loss curve is also exactly the textbook shape this time. No divergence, no dead ReLU, no implicit-regularisation surprises. Loss drops monotonically, accuracy climbs monotonically. **This is the first experiment in the entire sequence where the textbook diagram and the actual run agree perfectly.** That itself is a kind of milestone — it means I now have a baseline against which to recognise *non*-textbook behaviour in future runs.

## 4. Where the parameters of a CNN actually live

Counting the parameters of TinyCNN gives 105,866 — surprising in a couple of ways. Here is the breakdown:

| Layer | Parameter count | Share |
|---|---|---|
| `Conv2d(1 → 16, 3×3)` | 160 | 0.15% |
| `Conv2d(16 → 32, 3×3)` | 4,640 | 4.4% |
| `Linear(1568 → 64)` | **100,416** | **94.9%** |
| `Linear(64 → 10)` | 650 | 0.6% |
| **Total** | **105,866** | |

**The two convolutional layers, combined, account for less than 5% of the parameters.** The first fully-connected layer alone — the one right after `Flatten` — holds 95%. The 3×3 filters that do the actual *image understanding* are dwarfed in parameter count by the dense matrix that translates the flattened feature vector into 64 numbers.

A few things follow from this. First, **weight sharing in convolutions really is as efficient as advertised**. Conv2 has 32 filters, each of which is actually 3×3×16 = 144 weights (because the filter spans all 16 input channels — a detail I had been glossing over until I sat down to add up the parameters). That's 4,608 weights for the entire layer — for a layer that gets to look at every position of a 16-channel feature stack.

Second, **this exact discovery is one of the things that shaped modern CNN design.** Architectures from the 1990s through the mid-2010s — LeNet, AlexNet, VGG — all had this same shape: small convolutional trunk, enormous fully-connected tail. AlexNet famously had ~60M parameters, ~58M of which were in three FC layers, on top of a ~2M conv trunk. The FC tail wasn't doing significantly more useful work than the conv trunk, but it carried 95% of the parameters and 95% of the memory.

The fix the field eventually settled on is **Global Average Pooling**: instead of flattening a `32 × 7 × 7` feature stack into a 1568-long vector and then training a big FC, you average each 7×7 feature map down to a single number, getting a 32-long vector directly. The classifier becomes `Linear(32 → 10)` instead of `Linear(1568 → 64) → Linear(64 → 10)`. **Modern ResNets and friends use this trick — and that's a large part of why a ResNet-50 with millions of parameters can outperform an AlexNet with sixty million.**

This was a real "oh" moment for me. Before today I had been treating "number of parameters" as an indivisible bulk number that you compared between architectures. After today, "where in the architecture do those parameters sit" is the more useful question — and the answer says something concrete about the design choices a network reflects.

## 5. First look at information theory

The second half of today shifted from CNN to the foundations of *learned compression*, which is what the thesis area is really about. The starting question is naive-sounding: *what does it cost, in bits, to send a piece of information?*

**The Shannon answer.** A single event with probability *p* carries `−log₂(p)` bits of information. A guaranteed event (`p=1`) carries 0 bits — no surprise. A coin flip (`p=0.5`) carries 1 bit. Identifying one specific card out of 52 carries `log₂(52) ≈ 5.7` bits. Identifying one in a million carries `log₂(10⁶) ≈ 20` bits. The intuition is that each bit is a yes/no question that cuts the search space in half, and `−log₂(p)` is how many yes/no questions it takes to pin down an event of probability *p*.

**Entropy is average information.** For a random variable with multiple possible outcomes, the entropy is the probability-weighted average of `−log₂(p)` across all outcomes:

$$H(X) = -\sum_x p(x) \log_2 p(x)$$

A fair coin has `H = 1` bit. A fair 256-value byte (`p = 1/256` for each value) has `H = 8` bits — which is exactly why an 8-bit grayscale image stores 8 bits per pixel. A heavily skewed distribution like `[0.97, 0.01, 0.01, 0.01]` has `H ≈ 0.24` bits — almost zero, because almost all the time we already know what the value will be.

**Why this matters for compression.** Shannon's source coding theorem says: **the minimum average number of bits per symbol that any lossless encoder can achieve is exactly H(X)**. No clever code can do better; entropy is the floor. Gzip, PNG, JPEG-lossless, the entropy-coding stage in any video codec, the bit-budget reasoning in any learned compression paper — they are all operating under this floor.

This connects directly to the thesis area. Learned compression — the kind that papers like Ballé 2018, PCGCv2, HAC++, and my advisor's ICIP 2025 paper all build on — has the same shape:

1. Encode the data through a neural network into a **latent representation**.
2. Make that latent representation's distribution **as concentrated as possible** (low entropy) — that's what the network is trained for.
3. **Arithmetic-code** the latent representation, paying roughly H(latent) bits per symbol.

The `α · rate + β · distortion + γ · task` loss in my advisor's paper has a `rate` term, and the `rate` term is precisely **an estimate of the entropy of the network's latent representation**. Optimising rate means training the network to produce a latent whose distribution is as skewed as possible, so that fewer bits are needed to transmit it.

Before today, "rate" was a black box term for me — something I was supposed to nod at when reading the literature. After today, I can write down what it means in one sentence: *the entropy of the latent's distribution, in bits per symbol*. That is the bridge from foundations into the thesis area, and I crossed it this afternoon.

## 6. Stage 1 done

Four days, four blog posts, six lessons in the foundations. The map I drew on Day 1 had Stage 1 (deep learning basics) → Stage 2 (learned compression) → Stage 3 (3D / point clouds / 3DGS). As of tonight, **Stage 1 is closed**.

Stage 2 is the bridge to the thesis area: cross-entropy and KL divergence, then autoencoders, then quantisation, then arithmetic coding, then hyperprior, then end-to-end rate–distortion. By the end of Stage 2, the loss function in my advisor's paper should be transparent to me line by line. That is the *real* graduation line — not "all 15 lessons complete", but "I can read the methods section of the thesis area without papering over gaps".

Stage 3 is where Stage 2 meets the actual papers I made notes on a few weeks ago: point cloud representations, sparse convolution, 3D Gaussian Splatting, scalable compression, task-driven compression. Stage 3 happens in parallel with thesis work — by then I should be able to *read while doing*, not learn-then-do.

## Stuck points

By blog convention every post closes with the day's stuck points — what I got wrong and how I corrected it.

1. **A 3×3 filter is not always 9 weights.** I had been carrying around "9 weights and a bias" as the parameter count of a 3×3 filter. That's only true for a *1-channel* input. For multi-channel inputs (which is what every conv layer after the first one sees), a "3×3 filter" is actually `3 × 3 × in_channels` weights, because it has to span all the channels at once. Conv2 of my TinyCNN has filters that are 3×3×16 = 144 weights each, not 9. The 3×3 is the spatial part; the input-channel dimension is invisible in the kernel-size notation but very much there in the parameter count.
2. **Most CNN parameters do not live in the convolutional layers.** I had assumed the conv trunk would dominate the parameter budget. In a typical small-to-medium CNN, *the first fully-connected layer after Flatten holds 90%+ of the parameters* — in TinyCNN it was 94.9%. This is one of the historical observations that drove the field toward Global Average Pooling, and the reason modern ResNets can be vastly more parameter-efficient than older architectures.
3. **Entropy is not just "uncertainty".** I had been treating entropy as a soft buzzword for "how unsure am I about this random variable". The far more useful interpretation is: **the minimum average number of bits per sample that any lossless encoder can use, no matter how clever**. The compression-floor framing is what makes entropy useful in ML and compression; the "uncertainty" framing is a vague gesture in the same direction that doesn't actually carry the operational consequences.
4. **Skewed distributions compress dramatically.** A distribution like `[0.97, 0.01, 0.01, 0.01]` has 4 categories but only `H ≈ 0.24` bits of entropy — far less than the `log₂(4) = 2` bits that uniform 4-way data would need. That's a roughly 8× compression ratio attainable in principle, just by exploiting how skewed the distribution is. Every learned compression method is, at its core, training a network to make its latent representation as skewed as possible, so that arithmetic coding can ride the resulting low entropy down to a small bit budget.

*This is Day 4 — Stage 1 done. Stage 2 starts on Day 5: cross-entropy, KL divergence, and the loss I have actually been using for two days without fully understanding.*
