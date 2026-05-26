---
title: "Day 2 — A one-neuron network: training works, divergence is real, and overfitting did not show up the way I expected"
date: 2026-05-26 23:50:00 +0200
categories: [Foundations]
tags: [PyTorch, training, gradient-descent, learning-rate, overfitting, dead-relu, optimizer]
math: false
---

> Day 1 was concepts. Day 2 was code. The plan: write the smallest possible neural network end-to-end in PyTorch — one input, one output, one neuron — and use it to make every abstract idea from Day 1 actually concrete by watching it run on my own machine.
>
> The task I set the network: give it pairs `(x, y)` where `y = 2x + 3 + noise`, and see whether gradient descent can recover the rule. **A single-neuron `Linear(1, 1)` has exactly two parameters — one weight and one bias — and is mathematically identical to linear regression.** This is the simplest possible test bed for everything Day 1 talked about.

## 1. The training loop works exactly as advertised

The core of it is ten lines of PyTorch:

```python
model = torch.nn.Linear(1, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(2000):
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

The four lines inside the loop are the entire training algorithm:

1. **Forward pass** — predict.
2. **Compute loss** — measure how wrong (the "thermometer" from Day 1).
3. **`zero_grad` + `backward`** — compute the gradient of loss with respect to every parameter (this is PyTorch's autograd doing what Day 1's calculus would have been).
4. **`optimizer.step`** — nudge every parameter a tiny amount in the direction that decreases loss.

Repeat 2000 times and watch what happens. With random initialisation `weight = 0.7645`, `bias = 0.8300` (far from the true 2 and 3):

```
epoch    1   loss = 67.7597    weight = 1.5899   bias = 0.9846
epoch  200   loss =  0.1498    weight = 2.1511   bias = 2.1506
epoch 1000   loss =  0.0003    weight = 2.0062   bias = 2.9654
epoch 2000   loss =  0.0000    weight = 2.0001   bias = 2.9994
```

The network discovered `y = 2x + 3` to four decimal places, without anyone ever telling it the numbers 2 or 3. The whole signal it had was eight `(x, y)` pairs and the gradient of MSE. **Watching this on a screen is qualitatively different from reading about it.** It's the difference between "the network learns the rule" being a slogan versus a thing I've now seen happen.

But three things happened in the next hour that I hadn't quite expected. Each is a real lesson the Day 1 picture didn't fully prepare me for.

## 2. Bias converged much slower than weight — and the math says exactly why

Look at that table again. At epoch 200, `weight` was already 2.15 (close to the true 2), but `bias` was only 1.25 (far from the true 3). They moved at very different speeds even though they shared one and the same learning rate.

The reason is mechanical. SGD's update for any parameter is `−lr × gradient`. For MSE loss with one neuron `y = w·x + b`:

- The gradient of loss with respect to **weight** has an extra `x` factor: `∂L/∂w = (2/n) Σ x_i · (pred_i − y_i)`.
- The gradient with respect to **bias** has no such factor: `∂L/∂b = (2/n) Σ (pred_i − y_i)`.

With `x` averaging around 4.5 in my training data, the weight gradient ends up roughly 4.5× larger than the bias gradient on identical examples. The same `lr` therefore moves weight a lot and bias a little.

I'd absorbed *"a single learning rate moves all parameters uniformly"* as a fact. It isn't — the **effective** step size per parameter depends on the magnitudes of the inputs that feed into the gradient. In a real model with thousands of parameters connected to wildly different parts of the input, that means baked-in heterogeneous step sizes, even before you add anything fancy. This is one of the practical reasons modern optimizers like **Adam** exist: they keep a separate adaptive learning rate per parameter precisely so that small-gradient parameters don't get stuck while large-gradient ones overshoot.

## 3. Cranking `lr` from 0.01 to 0.5 produced textbook divergence in 14 epochs

Day 1 said *"learning rate too large → step over the valley → loss climbs"*. I reset the network to the same initial weights and tried `lr = 0.5`. Here is what the first 14 epochs actually look like:

```
epoch  1   loss =       67.76   weight =      +42.03
epoch  2   loss =    42902.96   weight =    -1003.85
epoch  3   loss = 27462286.00   weight =   +25455.90
epoch  4   loss = 17578866688   weight =  -643986.88
...
epoch 14   loss =         inf   weight = -6.9e+18
```

Two patterns worth committing to memory:

1. **The sign of weight flips every single step.** Each update overshoots the valley floor by more than the previous one, lands on the opposite slope where the gradient now points back the other way, and gets thrown across again. Pure ping-pong.
2. **The magnitude grows ~25× per step.** `0.76 → 42 → 1003 → 25456 → 643987 → …` — that's exponential divergence. Floating-point overflow ("inf") happens at epoch 14.

The clean mental picture from Day 1 is correct, but watching the actual numbers makes the *speed* of failure vivid in a way the picture doesn't. There is no "train longer" recovery from this — the network reaches infinity in 14 epochs.

## 4. Trying to demo overfitting did not give me the textbook picture — but gave me something more useful

The plan for this section was the textbook setup: train a small `Linear(1, 1)` and a deeper MLP on the same noisy data, watch the big network memorise training noise while the small one finds the underlying rule. Standard Day 4 material. What actually happened, in two attempts:

### Attempt 1 (3 training points, large noise σ=2)

Both models failed. The three noisy points happened to all sit near `y ≈ 10` (noise on each one pulled toward the same region), so any model trained on them naturally produced near-constant predictions around 10. The big MLP memorised the three points; the small `Linear(1, 1)` found the best-fit line through them, which had slope ≈ 0.14 — far from the true slope of 2. Both networks were "correct" relative to their data. **The data itself simply did not carry enough signal about the true rule.** This is not overfitting versus underfitting — it's *garbage in, garbage out*. A lesson Day 4 didn't have a name for.

### Attempt 2 (10 training points, small noise σ=0.5)

The small `Linear(1, 1)` nailed it: it learned `y ≈ 1.948 x + 3.317`, almost exactly the true `y = 2x + 3`, with healthy generalisation (test loss 0.03, training loss ≈ noise variance 0.25 — *which is the theoretical lower bound* and a confirmation that nothing is wrong).

But the big MLP (`1 → 64 → 64 → 1`, ~4500 parameters) did not overfit either. It produced a **constant** prediction of 14.03 for every test point — and 14.03 happens to be exactly the mean of the training `y` values. **It had not trained at all.**

The diagnosis: **vanishing gradients / dead ReLU.** With two ReLU hidden layers, vanilla SGD, and a learning rate gentle enough for the linear data, the gradient signal couldn't propagate through the network from a random initialisation. A large fraction of ReLU units output 0 at init, their gradients are 0, the upstream parameters get no signal, and the network gets stuck outputting a constant — the average of the training targets, because that is the global minimum of MSE for any constant function.

This is historically a *famous* failure mode. It is precisely the failure mode that motivated **Adam** (per-parameter adaptive learning rates that rescue starved parameters), **batch normalisation** (keeps activations well-scaled), and **residual connections** (let gradients skip layers). Without those modern tricks, deep ReLU MLPs trained with plain SGD are surprisingly fragile, especially on small linear problems where the gradient signal at init is weak.

So I never produced the picturesque overfitting demo. What I got instead was something I think is more useful:

- **Bigger is not automatically better.** A 2-parameter linear model crushed a 4500-parameter MLP on linear data — by a factor of ~1500× in test loss.
- **More capacity does not reliably cause overfitting.** It can cause *failure to train*, which looks similar on a single test number (high test loss) but is a completely different disease.
- **The textbook overfitting picture in deep networks needs better optimizers to even appear.** On a linear problem with vanilla SGD, the deep network doesn't get the chance to overfit because it doesn't train enough to memorise anything.

The diagram from Day 4 — *train loss near zero, test loss high* — is still correct. But the road to actually seeing it on a real run is more textured than the diagram implies, and what shows up first on a sloppy deep MLP isn't overfitting; it's a network that never trained. Knowing that distinction matters: next time I see a model with high test loss, "did it actually train" is now the first question, not the third.

## Stuck points

By blog convention, every post closes with the day's stuck points — what I got wrong and how I corrected it.

1. **A single learning rate does not move all parameters at the same speed.** Per-parameter gradient magnitudes vary with the inputs they're attached to — in `Linear(1, 1)` on data with `x ≈ 4.5`, the weight's effective step size is ~4.5× the bias's. The "one learning rate" abstraction hides this, and it's a real reason adaptive optimizers like Adam exist.
2. **"Bad test accuracy" is not the same as "overfitting".** When the data itself is too noisy or too sparse to carry the signal, even a 2-parameter linear model has bad test loss — but that's not overfitting. The diagnostic for overfitting is *low train loss together with high test loss*. Low *both* is underfitting or insufficient data; high *both* is something else still.
3. **Bigger networks do not automatically overfit — they may fail to train at all.** A 4500-parameter MLP collapsed to outputting the mean of the training targets, because vanilla SGD couldn't propagate gradient through the ReLU stack from initialisation. This is the exact failure mode that motivated Adam, batch normalisation, and residual connections. If I'd diagnosed only on "high test loss" I would have mislabelled this as overfitting.
4. **The textbook overfitting picture takes work to produce.** On a linear regression problem with plain SGD, the picture from a deep MLP failed to appear — what appeared was either *garbage in / garbage out* or *dead-ReLU non-training*. The picture is real, but the demonstration of it needs either Adam or specific architectural choices to reliably reproduce.

*This is Day 2. More to follow — including a re-run with Adam, where I expect the deep network finally to train and the actual overfitting picture to appear.*
