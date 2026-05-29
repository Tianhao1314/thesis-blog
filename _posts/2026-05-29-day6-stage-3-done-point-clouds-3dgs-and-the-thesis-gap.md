---
title: "Day 6 — Stage 3 done: point clouds, 3DGS, and the shape of the thesis I'm going to write"
date: 2026-05-29 23:58:00 +0200
categories: [3D-and-point-cloud]
tags: [point-cloud, sparse-convolution, 3DGS, scalable-compression, task-driven-compression, Minkowski-Engine]
math: false
---

> Day 6 was the last day of the formal curriculum. Four lessons covered the move from 2D learned compression — which Day 5 closed out — into 3D and the specific area my thesis will live in: **point cloud representation, sparse convolution, 3D Gaussian Splatting, and scalable + task-driven compression**. By the end of the day, the gap in the literature I'd been gesturing at since the very first day — "3DGS compression that is also scalable and also task-driven" — had become a concrete shape rather than a slogan.
>
> This is the post that closes the *learning* phase. From Day 7 onward the blog shifts in form. Instead of "I read about X and here's what I now understand", the posts will become "I read paper X and here is what it does and where it gets stuck", and eventually "I ran experiment Y and here is what came out". The thesis-work mode begins.

## 1. Point clouds — why the CNN toolkit fails on them

Image data is regular, dense, ordered, and fixed-size: a 256×256 image is a grid of exactly 65,536 pixels, each at a known address. Every operation in the CNN toolkit — `Conv2d`, `MaxPool2d`, `Flatten`, `BatchNorm2d` — quietly assumes all four of those properties.

A **point cloud** has none of them. A LiDAR frame is just a list of `(x, y, z)` triples — sometimes with attached `(r, g, b)`, `(normal_x, normal_y, normal_z)`, or intensity. The 3D positions are not on a grid; they're scattered at whatever locations the sensor happened to hit something. A typical frame has anywhere from 8,000 to 300,000 points, and the count varies frame to frame. There is no inherent "first point" — shuffling the list gives the same scene. And the points fill perhaps 1–3% of the actual 3D volume they live in; the rest is empty space.

Each of these breaks one of the CNN assumptions:

- **Unordered.** Point #17 and Point #142 can be swapped freely; the model must produce the same output. CNN's `Linear` and `Conv` layers are very much *not* invariant to input permutation. The first major fix was **PointNet** (2017), which used a symmetric aggregation (typically max-pooling) to summarise points in a way that doesn't care about ordering.
- **Variable size.** A `Linear(N, M)` layer needs a fixed `N`. Point clouds don't have a fixed point count. You either sample to a fixed number, build a variable-size network, or move to a sparse-grid representation.
- **Sparse.** If you voxelise a point cloud onto a 256³ grid, only 1–3% of voxels are occupied. A standard `Conv3d` does 7+ billion multiply-add operations on a layer at this resolution; ~99% of that computation is on empty voxels and is wasted.
- **Irregular.** The "3×3 neighbourhood" — a concept so basic in image CNNs that it disappears into the code — is genuinely not defined for raw point clouds. Neighbours don't sit at equal distances, neighbour counts vary point to point, and the local geometry can be very different at different parts of the cloud.

There are four standard ways to wrestle a point cloud into a neural network: keep the raw points and use a PointNet-style architecture, voxelise and use a sparse 3D CNN, build an octree and use specialised tree operations, or convert to a mesh and use graph-network methods. The two that matter most for compression are **voxelisation + sparse 3D CNN** (the path PCGCv2 and most modern point-cloud compression papers take) and **octree representations** (the path the MPEG G-PCC standard takes).

## 2. Sparse convolution — the engineering bridge that made everything possible

This was the lesson that genuinely surprised me. I'd been carrying around "sparse convolution" as a phrase that meant something like "a more efficient 3D conv". The reality is that it is **the bridge without which the entire 2D learned-compression toolkit would not transfer to 3D at all**.

The setup is the same as before: voxelise a point cloud onto a 256³ grid. With dense `Conv3d`, even a single 3×3×3 layer with 16 output channels takes about 7 billion multiply-add operations and the input tensor alone is 64 MB. Stacking five such layers blows past the memory budget of even modern GPUs.

Sparse convolution drops the dense storage entirely. Instead of an array indexed by `(i, j, k)`, the representation is a hash table from "occupied coordinate" to "feature vector":

```
coords   = [(i₁, j₁, k₁), (i₂, j₂, k₂), …]     ← N occupied positions
features = [f₁,           f₂,            …]     ← matching feature vectors
```

`N` is the actual number of occupied voxels — tens of thousands, not millions. A 3×3×3 convolution at coordinate `(i, j, k)` looks up neighbours `(i±1, j±1, k±1)` in the hash table; if a neighbour is occupied, its feature contributes; if not, it contributes zero (no computation). The outer loop runs only over the `N` occupied voxels. End result: roughly **50–100× faster than dense `Conv3d`**, with proportional memory savings.

The library that makes this usable is **Minkowski Engine** (NVIDIA + Stanford, 2019). The PyTorch interface is almost the same as for 2D CNNs — `ME.MinkowskiConvolution` replaces `nn.Conv2d`, `ME.MinkowskiMaxPooling` replaces `nn.MaxPool2d`, and the rest of the training loop is unchanged. Once you accept that mental swap, **PCGCv2's architecture is essentially "Ballé 2018 in 3D"**: the same encoder/quantiser/hyperprior/entropy-coder skeleton I walked through on Day 5, with `Conv2d` swapped out for `MinkowskiConvolution`. That equation — *PCGCv2 ≈ Ballé 2018 + sparse conv* — is the simplest description of where modern point-cloud compression sits in the literature.

So sparse conv is not an optimisation tweak. It's the engineering breakthrough — *the* engineering breakthrough — that let the deep-learning + 3D research programme exist at all. None of the compression literature I'm about to read for the thesis would exist without it.

## 3. 3D Gaussian Splatting — a completely different kind of 3D primitive

The other half of Stage 3 was 3DGS, which has been the dominant 3D representation in graphics and vision research since mid-2023 and is the actual format my thesis work will operate on. The conceptual move from point clouds to 3DGS is bigger than it sounds.

A point in a point cloud carries roughly 6–10 numbers: position `(x, y, z)`, maybe colour `(r, g, b)`, maybe a normal. A point's job is to *sample a surface location* — many points together approximate the surface of an object.

A **3D Gaussian** in 3DGS carries roughly 50 numbers: position `(x, y, z)`, a 3D covariance specifying scale and rotation (4–7 numbers depending on parameterisation), opacity `α`, and a stack of spherical-harmonic coefficients describing colour from different viewing angles (typically ~45 numbers). A Gaussian's job is to *cover a small fuzzy region of space* with a view-dependent appearance. The scene is the alpha-blended composition of millions of these blobs.

The numerical consequence: for the same scene, where a point cloud might carry 0.5 M points × 8 numbers ≈ 4 M floats (≈ 16 MB), a 3DGS representation typically carries 5 M Gaussians × 50 numbers ≈ 250 M floats (≈ 1 GB). **Two orders of magnitude more data, for the same scene.** That's the brute fact that makes 3DGS compression a real-world problem rather than a synthetic one.

The reason 3DGS is bigger is also the reason it's better at rendering: the spherical-harmonic colour coefficients let each Gaussian look correct from many viewing angles, which is what gives the rendered output its photorealistic quality. Point clouds carry per-point colour that doesn't change with viewpoint, so they look fine when rasterised as splats but flat under proper lighting. 3DGS pays for the data volume with appearance fidelity.

What's compressed in 3DGS compression is exactly those 50-numbers-per-Gaussian. The literature has converged on five rough strategies:

- **Pruning.** Drop Gaussians that contribute little. (SA-3DGS does this.)
- **Quantisation.** Float32 → 8-bit or 4-bit per parameter. (Standard across most papers.)
- **Entropy coding with a learned context model.** A hyperprior, just like Day 5. (HAC++ uses a hash-grid hyperprior; this was the headline 2024 result.)
- **Parameter sharing across nearby Gaussians.** Spatial clustering or anchor points.
- **Scalable encoding.** A single bitstream that serves multiple quality levels. (GoDe, PCGS — both 2024.)

My advisor's ICIP 2025 paper sits in the intersection of all five plus a sixth direction (task-driven compression — see next section), and it's exactly that intersection that the thesis area will live in.

## 4. Scalable and task-driven compression — the advisor's frontier

The final lesson of the day covered two ideas that name what my advisor's work is actually about. Both are now legible to me as concrete technical choices, not just slogans.

**Scalable compression** is the idea that *one* bitstream should serve users with different bandwidth or quality needs, rather than storing separate copies at each quality. It works by structuring the bitstream into a small **base layer** plus successive **enhancement layers**. A decoder that reads only the base layer reconstructs a coarse version of the scene. A decoder that reads base + enhancement 1 reconstructs a medium version. A decoder that reads everything reconstructs the full quality. JPEG progressive, H.264 SVC, and JPEG2000 all do this in 2D; doing it for 3DGS specifically is roughly an 18-month-old research direction.

**Task-driven compression** is the more conceptually interesting of the two. Traditional compression optimises for `||x − x'||²` — *how visually faithful is the reconstruction*. Task-driven compression optimises for *how accurately a downstream machine task performs on the reconstruction*. The loss looks like:

$$\mathcal{L} = \alpha \cdot \text{rate} + \beta \cdot \text{distortion} + \gamma \cdot \text{task}$$

The `γ · task` term measures how well a downstream classifier, detector, or segmenter performs on the decoded data. The whole pipeline — encoder, decoder, downstream model — is trained end-to-end. The empirical result, replicated across multiple 2D image-compression papers, is that for a fixed bit budget, task-driven encoders maintain much higher downstream accuracy than encoders trained purely on `distortion`.

This matters because **most of the data being transmitted in modern systems is consumed by machines, not humans**. Autonomous-vehicle LiDAR streams to a remote perception model; factory sensors feed an inspection AI; surveillance video gets analysed by automated event detectors. For all of these, optimising for human-visible reconstruction wastes a substantial fraction of the bit budget on detail that the downstream model neither uses nor needs.

When you combine the two ideas — scalable + task-driven — the result is **a single bitstream with a small "machine layer" that suffices for the downstream task, plus optional "human layers" that progressively add visual fidelity**. The machine consumer downloads only the first few KB; the human consumer downloads the rest.

There is, at the time of writing, no published work that does this *for 3DGS specifically and from end to end*. That is the gap. It's the gap my advisor's group is going to fill, and the gap my thesis is going to live in.

## 5. What I can now read

Stage 3 closes the formal curriculum. The list of papers from my 28-paper literature scan that I can now open and parse line by line:

- **PCGCv2** — *Ballé 2018 + sparse conv*. Reads cleanly.
- **HAC++** — hash-grid hyperprior for 3DGS. Reads cleanly.
- **CSGaussian** — rate-distortion + segmentation, the closest published work to my advisor's direction. Reads cleanly.
- **Bits-to-Photon** — end-to-end scalable point-cloud compression with a 3DGS-renderable bitstream. Reads cleanly.
- **GoDe, PCGS** — progressive / scalable 3DGS compression. Reads cleanly.
- **SA-3DGS** — self-adaptive 3DGS compression with importance-aware pruning. Reads cleanly.
- **The 3DGS compression surveys** (Bagdasarian et al., Yuan et al.) — both readable, and useful for putting all the above into one mental map.

This list is the *operational* meaning of "graduation". I can now read what my advisor and her collaborators read; I can talk to her in the language she works in; and I can pick a small piece of the remaining gap and start working.

## 6. The blog mode shift

From Day 7 onward, the posts here change in form. The lesson-by-lesson curriculum is done. The next mode of work is:

- **Paper deep-dives** — sit with one paper, work through what it does, where it succeeds, where it stops short. The 28-paper notebook I built before the curriculum becomes a series of public second-pass reads.
- **Code reproduction** — fork PCGCv2 from the official GitHub, get the baseline running on ModelNet40 or a similar dataset, sanity-check the numbers against the paper's reported results, and use that as a base for any experimental work going forward.
- **Thesis-topic conversations with my advisor** — set up the next meeting, walk her through what I now understand and the specific direction I want to take, and get her to refine the question I'll spend the next six months on.
- **Experiments** — by the end of Day 9 or so, this blog should have its first post that says "I ran X, here's what came out, here's what I now think it means".

The earlier posts were "I learned X today". The next posts will be "I tried X today, and here's what I think it shows". That's a real change in the kind of work being done, and the blog will reflect it.

## Stuck points

By blog convention every post closes with the day's stuck points — what I got wrong and how I corrected it.

1. **Point clouds and 3DGS are not the same kind of 3D representation.** I had been carrying them in my head as "two flavours of 3D point data". They are not. A point in a point cloud carries ~6 numbers and samples a surface location; a Gaussian in 3DGS carries ~50 numbers and covers a small fuzzy region of space with a view-dependent appearance. The data-volume difference for the same scene is roughly two orders of magnitude. Treating them as similar primitives misses why 3DGS files are huge and why 3DGS compression is a different research problem from point-cloud compression.
2. **Sparse convolution is not an optimisation, it is a bridge.** I had been mentally classifying sparse conv as "the version of 3D conv you use when you care about speed". The far more useful framing is that it's the engineering result without which **the entire learned-compression toolkit from Day 5 would not run in 3D at all**. PCGCv2 reads as "Ballé 2018 + sparse conv" precisely because sparse conv is what made the 2D-to-3D transfer of the framework operationally possible.
3. **Task-driven compression isn't about cleverer bit-packing — it's about changing the optimisation target.** I had been imagining task-driven methods as something like "use the downstream task as a regulariser on top of visual quality". The actual move is more radical: replace `‖x − x'‖²` with `Task_Loss(x')` as the dominant term in the loss. The pipeline is trained end-to-end with the downstream model in the loop. The empirical result that "fewer bits are needed for the same downstream accuracy" is *not* magic; it's that you're no longer paying bits for fidelity the downstream model doesn't use.
4. **The thesis "gap" has a specific shape.** I had been carrying the gap in my head as "3DGS compression but for AI tasks", which is too vague to start working on. The more precise statement is: **scalable + task-driven + 3DGS** — a single bitstream whose base layer is sized for downstream machine perception, with optional enhancement layers for human viewing, applied specifically to 3DGS rather than to 2D images or to point clouds. No published end-to-end work does this yet. That is the specific question to pose to my advisor at the next meeting, not the vague version.

*This is Day 6 — the formal curriculum closes here. Day 7 starts as the first paper deep-dive: Bits-to-Photon, the paper that sits closest to the thesis-shaped gap.*
