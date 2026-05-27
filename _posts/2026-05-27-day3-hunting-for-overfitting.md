---
title: "Day 3 — Three attempts to demonstrate overfitting, and the three different lessons each one taught instead"
date: 2026-05-27 15:30:00 +0200
categories: [Foundations]
tags: [PyTorch, overfitting, generalization, dead-relu, adam, optimizer, implicit-regularization]
math: false
---

> Day 2 ended with a promise: the deep MLP that failed to train under vanilla SGD would, in principle, train fine under Adam — and then the textbook overfitting picture should finally appear. I sat down today to make it happen. It did not, and what happened instead taught me a lot more than the textbook picture would have.
>
> This post is the honest log of three attempts to demonstrate overfitting — none of which produced a textbook overfitting curve. Each one failed in a different way, and each failure turned out to be a real, named phenomenon in modern ML practice. I think the resulting picture is more useful than the diagram I was originally chasing.

## The setup I kept tweaking

The experiment was always the same shape: a true rule `y = 2x + 3`, training data made by sampling that rule with added Gaussian noise, and two models — a small one with the right capacity (`Linear(1, 1)`, 2 parameters) and a large one that's massively overparameterised for the task (`Sequential(Linear(1, 64), ReLU, Linear(64, 64), ReLU, Linear(64, 1))`, ~4500 parameters). Train both on the same data, evaluate both on a clean test grid, see whether the big network memorises the noise while the small one recovers the line.

What changed across attempts was the noise level, the number of training points, and the optimiser. The textbook expectation was always the same: training loss → near zero on the big network, test loss high, predictions wiggle wildly between training points, a clean *train low + test high* gap as a diagnostic. I never got that picture. Here is what I got instead.

## Attempt 1 — Both models fail, and neither failure is overfitting

**Setup:** 3 training points (`x = 2, 4, 6`), noise standard deviation σ = 2.0, SGD optimiser. Aggressive overparameterisation (1500× more parameters than training samples) plus heavy noise should, on textbook reasoning, give the big network plenty of room to overfit and the small one a chance to recover the underlying line.

**What I got:** Both models had high test loss, but in different ways. The small `Linear(1, 1)` learned `y = 0.14 x + 9.82` — wildly wrong (true slope is 2). The big MLP collapsed into nearly-constant predictions around `y ≈ 10.4`, fitting the three noisy points well but extrapolating flat.

**What actually happened:** the noise sample for this seed put all three training points near `y ≈ 10`. Specifically, with `x = 2, 4, 6` and true `y = 7, 11, 15`, the noise pushed them to `y ≈ 10.08, 10.41, 10.64`. The "best line through these three noisy points" has slope ≈ 0.14, not 2 — because the three noisy observations themselves don't carry the true slope. The small linear model dutifully learned the wrong line. The big network memorised the three near-flat points and extrapolated flat between and beyond them.

**Lesson:** *Garbage in, garbage out.* When training data is too sparse and too noisy, the *data itself* doesn't carry the signal. No amount of model-capacity tuning can recover what the data doesn't contain. This is **not** overfitting versus underfitting — it's a third category that Day 4 of my foundations notes didn't even name: insufficient signal. The diagnostic for overfitting is *low train loss + high test loss*; here both losses were high relative to the truth, but only because the truth was unreachable from this data.

What I had been hoping to see (a textbook overfitting gap) wasn't even a coherent question to ask, because the data didn't support recovering the truth in the first place.

## Attempt 2 — The small network nails it; the big network never trains

**Setup:** Bumped the training set to 10 points (`x = 1, 2, …, 10`), kept SGD, dropped σ from 2.0 to 0.5. Now the data is plentiful enough and clean enough that the true rule should be recoverable. Run both models, expect the small one to find the line and the big one to memorise the noise around it.

**What I got:** The small `Linear(1, 1)` recovered the rule almost perfectly — `y = 1.948 x + 3.317` against the true `y = 2x + 3`, with train loss ≈ 0.25 (matching the noise variance σ² = 0.25, the theoretical lower bound), test loss ≈ 0.03, and predictions on a dense test grid that traced the true line to within ±0.2 across the range. Day 4 textbook behaviour. ✅

The big MLP, on the other hand, output a constant: `y ≈ 14.03` for every input from 0 to 10. Training loss `≈ 31.55`, test loss `≈ 37.74`, predictions literally the same number regardless of input. The flat value 14.03 happens to be precisely the mean of the training `y` values.

**What actually happened:** **dead ReLU + vanishing gradients.** At random initialisation, a meaningful fraction of the 128 ReLU units in the network output exactly 0 for some inputs, contributing 0 gradient. The upstream parameters got no signal. The optimiser couldn't move them. The only function the network could express under those conditions was the constant function — and the constant that minimises MSE against any training set is the mean of the training targets. So the network "converged" to that mean and stayed there.

The training loss number, 31.55, is variance of the training `y` values around their mean — which is also exactly what you'd compute if you predicted the mean of the targets for every input. That's the diagnostic that says "this model isn't a model, it's a histogram."

**Lesson:** *Bigger is not automatically better — sometimes it doesn't train at all.* Deep ReLU MLPs trained with plain SGD on small problems are surprisingly fragile to initialisation. This is historically the failure mode that motivated **Adam** (per-parameter adaptive learning rates), **batch normalisation** (keeps activation magnitudes well-scaled across layers), and **residual connections** (let gradients skip past dead regions). Without those modern tricks, the picture you get from a deep ReLU MLP isn't overfitting — it's no fitting at all. The high test loss looks like the same symptom as overfitting if you only look at one number, but it's a completely different disease and requires opposite treatment.

I had walked in expecting *train loss → 0, test loss → high*. What I got was *train loss → variance of targets, test loss → variance of targets too*. The big network never even got the chance to overfit.

## Attempt 3 — Adam fixes the training problem; the overfitting still doesn't appear

**Setup:** Identical to Attempt 2, but replaced the SGD optimiser with Adam. Adam keeps an adaptive per-parameter step size — small-gradient parameters get amplified, large-gradient parameters get dampened, so the heterogeneous gradient magnitudes that killed plain SGD no longer block training. With Adam, the deep MLP should train fine, and *then* the textbook overfitting should appear.

**What I got:** Adam definitely fixed the training problem. The big network's training loss dropped from SGD's 31.55 to **0.1181** — a ~270× improvement. Dead ReLU was real, Adam routed around it.

But the big network's *test* loss came in at **0.1101** — slightly *lower* than its training loss. Test-to-train ratio ≈ 0.9. The predictions on the dense test grid traced the true line `y = 2x + 3` to within roughly ±0.3 across the entire range, including between training points. The big MLP didn't memorise the noisy points and wiggle wildly between them; it found a smooth function that's essentially indistinguishable from what the small linear model found.

This is the opposite of overfitting. With 4500 parameters and only 10 training points — i.e. a model with 450× more capacity than the training set — the network *generalised better than its training-set fit would suggest*.

**What actually happened:** this is a real, named, and still-debated phenomenon in modern deep learning. It goes under names like **implicit regularisation**, **benign overfitting**, and (when measured carefully across model sizes) **double descent**. The empirical observation, replicated across many domains over the last several years, is that **overparameterised neural networks trained with SGD or Adam often converge to "low-complexity" solutions** — not the wiggly memorising solutions that classical statistical learning theory says should dominate. The optimiser's dynamics on the loss landscape happen to favour the smooth, simple solutions that also happen to generalise well. There are theoretical explanations (norm-minimising bias of gradient flow, kernel-regime analyses, etc.), but it's still a frontier area; the gap between what classical theory predicts and what modern networks actually do is large enough that explaining it is an active research programme.

**Lesson:** *Overparameterisation alone does not produce overfitting in modern deep learning the way the textbook picture suggests.* Plain SGD or Adam contains its own implicit regularisation, and the kind of dramatic train-low-test-high gap that motivated the four classical anti-overfitting techniques (more data, L2 regularisation, dropout, early stopping) doesn't reliably appear even when you set up the situation to provoke it. The classical picture isn't *wrong* — it's just an incomplete description of what overparameterised networks actually do, and getting an overfitting curve on demand requires more aggressive setups (much noisier data, much fewer training points, much longer training) than the textbook suggests.

## What the three lessons add up to

I wanted a clean overfitting demonstration. I now have three failure modes, none of which is overfitting:

| Setup | Symptom | Underlying cause |
|---|---|---|
| σ = 2.0, 3 points, SGD | Both networks fail with high test loss | Insufficient signal in data |
| σ = 0.5, 10 points, SGD (deep) | Big network outputs the mean of training targets | Dead ReLU / vanishing gradients |
| σ = 0.5, 10 points, Adam (deep) | Both networks generalise; no overfitting gap | Implicit regularisation of Adam on overparameterised networks |

In each case, "high test loss" alone would have been the wrong diagnostic. The diagnostic that actually distinguishes these is *the shape of the train-loss curve* combined with *what the model's predictions look like*. The first failed because both losses were high. The second failed because both losses were high *and predictions were a constant*. The third "failed" (relative to my expectation) because both losses were low and roughly equal — the picture-perfect generalisation case, just from a model that "shouldn't" generalise according to one school of theory.

This is, I think, a much more useful map of the territory than the textbook diagram alone would have given me. When I see "test loss is bad" in a future experiment, I now have at least three named hypotheses to consider before reaching for the four classical anti-overfitting remedies — *data has no signal*, *network isn't training*, *generalisation is fine and something else is wrong* — and a sense of how to tell them apart from the train loss and the predicted curve.

The next attempt will be the textbook demo done deliberately and artificially: σ = 1.5 noise, only 5 training points, Adam, many more epochs. I expect that one to finally show the train-low-test-high wiggle gap — but as a contrived setup, not a default outcome of overparameterisation.

## Stuck points

By blog convention every post closes with the day's stuck points — what I got wrong and how I corrected it.

1. **"High test loss" is not a sufficient diagnosis.** I had been treating *high test loss* as synonymous with *overfitting*. It isn't. The three runs above all produced high test loss for completely different reasons — insufficient data signal, untrained model, expected statistical noise — and each requires opposite treatment. The diagnostic that actually distinguishes them is the *combination* of train loss, test loss, *and what the predicted function looks like*.
2. **Vanilla SGD on deep ReLU MLPs is far more fragile than I'd assumed.** I had been treating SGD as "the basic case" and Adam as "an upgrade". On a small toy problem with a 3-layer ReLU MLP, plain SGD didn't just train slowly — it failed to train at all, collapsing to a constant. The historical reasons Adam, BatchNorm, and ResNet exist are real; they're not optional accelerators, they're often necessary for the model to leave its initial state.
3. **Overparameterisation does not automatically cause overfitting in modern deep learning.** Classical statistical learning intuition predicts that a 4500-parameter model fit to 10 data points should memorise the noise and generalise poorly. It didn't. With Adam + ReLU + small noise, the network found a smooth function close to the true line — train loss ≈ 0.12, test loss ≈ 0.11. This is the implicit-regularisation / benign-overfitting phenomenon that's been a major research thread in deep learning theory over the last decade, and it means the classical "regularise to prevent overfitting" advice doesn't always apply.
4. **The textbook overfitting picture takes deliberate effort to construct.** It is real and important to understand, but it is *not* the default behaviour of overparameterised networks. To see it cleanly I'll need a deliberately adversarial setup (more noise, fewer training points, much longer training). That's worth doing — but the more important lesson is that, in the regime that's actually used in modern ML practice, the picture often doesn't appear.

*This is Day 3. The next post will go after the textbook overfitting curve deliberately, then move on to convolutional networks.*
