---
title: "Day 1 — Foundations refresher: what deep learning is, how it learns, why it overfits"
date: 2026-05-26 15:42:00 +0200
categories: [Foundations]
tags: [deep-learning, neural-network, training, loss, gradient-descent, overfitting]
pin: false
math: false
---

> This is the first post on this blog and the Day 1 of my master's-thesis process.
>
> My thesis area is **scalable, task-driven learned compression of point clouds and 3D Gaussian Splatting**. I have used PyTorch in a previous sEMG impedance project (LSTM, MC-Dropout, attention), but enough of that code rested on architectural priors I had taken on faith that I want to put the basics back on first-principles ground before reading deeper into the compression literature.
>
> Writing convention here: the body is a walk-through of the concepts, but I deliberately keep the moments where I caught myself getting something *wrong* — for a process journal, those stuck points are the most useful parts to preserve.

## 1. What deep learning is, mechanically

The way I had been thinking about programming for a long time: *figure out the rule, write it as code, the program executes it.* If you wanted a program to classify cats vs. dogs, you would sit down and write out the rule yourself — pointy ears, whiskers, fur texture — and turn it into a stack of `if/else`s.

Deep learning runs the **opposite** direction. It doesn't ask you for the rule; it asks you for **many labelled examples**. You give it tens of thousands of photos tagged *cat* or *dog*, and **it** infers from those photos what cats and dogs actually differ on.

> **One-line summary.** You hand it the examples; the machine hands back the rule.

The thing that took me longer to accept than I'd like to admit was that "the machine learns the rule by itself" sounds vaguer than it really is. Concretely, the rule lives, very specifically, inside a few hundred thousand to a few hundred million numbers — the **weights** — and the entire learning process is just adjusting those numbers until they're "right". The next three sections are about what the weights are, and how they get adjusted.

---

## 2. What a neural network actually looks like

A neural network sounds mystical, but structurally it's just **many very simple units, wired together**.

### One neuron: the smallest unit

The smallest unit is a **neuron**. What it does is almost embarrassingly simple:

**It takes a few input numbers, gives each input a "weight", sums them, and outputs a single score.**

A toy example. Suppose I'm deciding *should I go to the beach today?* In my head I'm essentially running one neuron:

- Input 1: temperature (28 °C)
- Input 2: is it raining (0/1)
- Input 3: is it the weekend (0/1)

These three signals are **not equally important** to me. That "importance" is the **weight**. A neuron multiplies each input by its weight, sums them, and out comes a score.

Crucially, a weight has not just a magnitude but a **sign**:

- A **positive weight** pushes the score up — toward "yes, go". A comfortable temperature gets a positive weight.
- A **negative weight** pushes the score down — toward "no, don't go". Rain gets a negative weight, and a fairly large one in magnitude.

> The single line that ends up underwriting everything else: **the weights in a neuron are not set by a human — they are learned by the machine.** When section 1 said "the machine hands back the rule", the rule *is* those numbers.

**Stuck point (1).** I had been carrying around the implicit picture that a "more important" feature gets a larger weight. Working through the beach example forced me to admit that *suppressing* a decision is just as legitimate as supporting it, and that the way a network represents "this feature pushes against the answer" is a weight with negative sign and large magnitude. Trivial in hindsight; not how I had been visualising it.

### A layer: many neurons looking at the same input

One neuron alone is limited — it can only make one simple decision. Fine for the beach. Not fine for "is this photo a cat".

The fix is to **stand many neurons side by side, all looking at the same input, but with different weights**. That row of neurons is called a **layer**.

For a cat photo, in the same layer:

- Neuron A's weights might make it sensitive to **pointy ears**.
- Neuron B's weights might make it sensitive to **whiskers**.
- Neuron C's weights might make it sensitive to **fur texture**.

They're looking at the same pixels, but because the weights differ, each one is **chasing a different cue**. The output of a whole layer isn't a single score anymore — it's a list of scores, a kind of "cue inventory" of the image.

### Stacking layers: this is the "deep" in deep learning

But **one layer, staring directly at raw pixels, can only catch the simplest things** — a short edge, a brightness boundary, a patch of brown. Abstract concepts like "how many legs" or "one eye" are well beyond what a single layer can extract.

The fix: **stack layers, and let each layer look at the previous layer's output instead of at the original pixels.**

- **Layer 1** reads raw pixels → outputs simple cues ("vertical edge here", "brown patch there").
- **Layer 2** no longer sees pixels — it sees **Layer 1's cue list** → composes short edges into small shapes ("these few edges form what looks like a leg").
- **Layer 3** sees **Layer 2's list** → composes shapes into bigger judgments ("four legs, dog-like body").
- **Final layer**: synthesise everything → "this is a dog".

Each layer's input vocabulary is literally the previous layer's output. Cues compose from simple to complex.

> **"Deep" in deep learning is the layer count.** Many layers is what lets the network compose simple cues into progressively more complex concepts — it's why a single very wide layer cannot make the jump from "pixels" to "how many legs" directly.

**Stuck point (2).** I had been treating "deep" as marketing-speak — as if it were just a flashy adjective. It isn't. *Deep* is literal: the layer count is large, and the consequence is a strict feature ladder where each layer's primitives are the previous layer's outputs. Saying "the network learns hierarchical features" landed for me only after I drew the ladder out by hand.

---

## 3. How the machine actually learns the weights

When the network is first built, all weights are **random numbers**. Feed it a cat photo at this point and the output is essentially a guess.

**How does the machine adjust from "random" toward "correct"?**

### Step 1: build a thermometer — the loss function

For the machine to move in any useful direction, it first needs a measurement of "how bad am I right now".

The **loss function** is that thermometer. It takes two things:

- **Input**: the network's current answer + the correct answer.
- **Output**: a single number measuring "how wrong was that".

The rule is simple: the more wrong, the larger the number; the more right, the smaller; perfect equals zero.

**Why must this be reduced to a single number?** Because only numbers compare cleanly and only numbers can be optimised. "More wrong or less wrong" as a feeling is not computable; a specific number is.

In real training we don't compute the loss on one example at a time — we average the loss across a batch of examples (say a thousand). That average becomes the single number we are trying to drive down. **The entire goal of training reduces to: shrink this number.**

A small mnemonic that I find sticks: **confidently right < hesitantly right < hesitantly wrong < confidently wrong**, loss ascending. The worst case for any loss function is confidently giving the wrong answer.

### Step 2: blindfolded descent — gradient descent

Here comes the part that does most of the work — *how* does the machine adjust the weights based on the loss number?

The mental picture I keep coming back to: **imagine you've been blindfolded and dropped onto a mountain, and your job is to find the lowest valley.** You can't see anything, but you can feel the slope right under your feet. The strategy is obvious:

1. Feel which direction is most steeply downhill.
2. Take a small step that way.
3. Re-feel, repeat.

**Gradient descent is exactly this.** Translating the metaphor:

| Metaphor | Neural network |
|---|---|
| Your position on the mountain | The current values of all the weights |
| Your altitude | The current loss |
| The slope under your feet | "If I nudge this weight, how does the loss change" — this is the *gradient* |
| Taking one step | Adjusting every weight a tiny amount in the direction that reduces loss |

Repeat hundreds of thousands or millions of times. The loss descends step by step; the network gets steadily more accurate.

### Step 3: how big a step — the learning rate

The step size matters more than it first looks:

- **Too small**: direction correct, progress glacial. You might take months to reach the valley.
- **Too large**: you overshoot the valley floor in one step and land on the opposite slope, *higher* than where you started. Then you overshoot back. Loss bounces around without converging.
- **Just right**: small, steady descent.

That step size is called the **learning rate**. It's a number you set before training (typically small, on the order of 0.001). Setting it badly is one of the easiest ways to make a perfectly good network refuse to train; setting it well converges in minutes. Tuning it is a craft I'll learn properly when I start writing PyTorch code.

> **The whole training story in one line.** Weights start random → the loss function turns "how wrong" into a single number → gradient descent walks downhill on that loss surface → learning rate sets the step size → repeat many times.

**Stuck point (3).** "Machines self-adjusting their own weights" felt magical to me the first few times I heard it. It isn't magical at all — it is **multi-variable function minimisation**, the same thing one meets in calculus, except the dimension count is in the millions and we use the mechanical blindfolded-descent procedure instead of solving "derivative = 0" analytically. Once I named it correctly, the mystery dissolved.

---

## 4. Trained ≠ actually useful

This is the section that drags theory back to reality.

### The most embarrassing failure mode: 99% train, 60% test

You follow section 3, your loss drops, and your network is now 99% accurate on the photos you trained it on. You hand it a **new** cat photo (one it has never seen) and it confidently tells you it's a dog. You try ten new photos and it gets six right — 60% accuracy. From 99% to 60%, just by changing the photos.

What went wrong has a name: **overfitting**.

A student analogy I keep using:

- **Student A** memorises the answers to the practice problems. "Problem 17 is C, problem 18 is A, problem 19 is B." Perfect on the practice set. New problems on the real exam — disaster.
- **Student B** works to understand the *method* behind each problem. Only 85% on the practice set, but the real exam goes fine.

**Overfitting is the network being Student A.** When training data is limited and the network is large, it has enough "memory capacity" to literally memorise each training image — down to the pixel noise — instead of learning what cats look like in general. Any new image, with different noise, falls through.

### The diagnosis needs *two* numbers

Looking only at training accuracy will mislead you (Student A also has perfect practice scores). The trick is to set aside a **test set** that the network never sees during training, and then check after training is done.

| Train accuracy | Test accuracy | Diagnosis |
|---|---|---|
| High | High | ✅ Healthy |
| High | Low (big gap) | Overfitting (memorising) |
| Low | Low (both bad) | Underfitting (didn't learn at all) |

**You need both numbers. Either one alone is not enough.**

**Stuck point (4).** Given a hypothetical 70%-train / 68%-test model, my instinct was to call it overfitting "because both numbers look bad". That's wrong. Both numbers being **low and close together** is *underfitting* — the model didn't learn the material, full stop. It is a completely different disease from overfitting, and the treatments are opposite (underfitting wants more capacity or more training; overfitting wants regularisation, more data, or shorter training). Mis-classifying these two would have wasted a lot of time later in a real experiment.

### Four remedies for overfitting

| Remedy | Student analogy | Mechanism |
|---|---|---|
| More data | Practice set grows to 50,000 problems | Memorising all of them becomes more costly than learning the actual rule |
| Regularisation (L2 / weight decay) | "Your answer sheet can't be longer than three lines" | Penalises large weights; forces the network toward a simpler, more general rule |
| Dropout | Random actors take the day off in rehearsal | Forces every neuron to be useful — the network can't lean on a few "I memorised example 247" specialists |
| Early stopping | Watch both numbers and stop at the turning point | Halt training before the network starts to memorise rather than learn |

**Early stopping has one extra wrinkle.** It needs to *monitor* a number during training. If you use the test set for that monitoring, you've effectively let the training procedure peek at the test set, and the test set is no longer an honest measure of final quality. The standard fix is to split a third set — the **validation set** — out of the training data:

- **Train set** (~70%): what the network actually learns from.
- **Validation set** (~15%): checked repeatedly during training; this is what early stopping watches.
- **Test set** (~15%): touched **exactly once**, after training is fully done, to produce the final reported number.

In practice all four remedies are used together, not as alternatives.

---

## 5. Day 1 wrap-up — what this unlocks

After Day 1 I can talk to my advisor using the actual vocabulary — "end-to-end", "loss", "training vs. test", "overfitting", "learning rate" — and have each word point to something concrete rather than to a vague impression.

I can also start reading some of the papers I previously bounced off of. The opening of *Learned Point Cloud Compression for Classification*, which talks about end-to-end joint loss, finally parses: the "loss" is the thermometer from section 3; "joint" means several thermometers averaged together with weights. That kind of sentence used to be a wall of vocabulary; now it is a sentence with referents.

**Day 2 will pick from two paths:**

- **Path A**: hand-write a tiny PyTorch network and actually train it, to feel the learning-rate and overfitting points instead of just reading them.
- **Path B**: move into CNNs (convolutional neural networks) — the specific architecture family that the point-cloud and 3DGS compression literature builds on top of.

My inclination is **A first, then B**. Having trained something with my own hands once will make the CNN section land much more concretely.

---

## Stuck points (summary)

By blog convention every post closes with the day's stuck points — what I got wrong and how I corrected it:

1. **Negative weights** — I was carrying the "bigger weight = more important" picture, until the beach example forced me to realise weights must be allowed to be negative (suppression is just as legitimate as support).
2. **"Deep" is not a metaphor** — I had been reading *deep* as a flashy adjective; the layer count is literal, and each layer's input vocabulary is the previous layer's output, all the way up the feature ladder.
3. **Overfitting ≠ underfitting** — both diseases produce "bad accuracy", but one comes from memorisation and the other from never learning; the diagnostic depends on the train-vs-test *gap*, not on the level of either number alone.
4. **Gradient descent demystified** — the "machines adjust their own weights" framing sounded magical; once I named it as multi-variable function minimisation, the magic went away and I could think about it the way I'd think about any optimisation problem.

*This is Day 1. More to follow.*
