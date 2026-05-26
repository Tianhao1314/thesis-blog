# Thesis-in-Progress

Tianhao Wu's master's-thesis preparation and writing log.

**Live site (once deployed):** https://tianhao1314.github.io/thesis-blog/

Thesis direction (set by my advisor [Xiumei Li](https://www.lms.tf.fau.de/faudir/xiumei-li/), FAU Erlangen–Nürnberg):
**scalable, task-driven learned compression of point clouds and 3D Gaussian Splatting** for joint human–machine use.

## What this is

A working journal — not a research showcase. Daily notes on what I'm reading, what concept I worked through, what experiment I ran, where I got stuck, how I climbed out. The research showcase lives at [tianhao1314.github.io](https://tianhao1314.github.io).

## Writing a new post

```bash
# 1) Create a file in _posts/. Filename convention (Chirpy requirement):
#    YYYY-MM-DD-slug.md
#    e.g. 2026-05-27-day2-pytorch-hello.md

# 2) Front matter template:
---
title: "Title here"
date: 2026-05-27 20:30:00 +0200
categories: [Foundations]   # or: Learned-compression, 3D-and-point-cloud, Paper-deep-dives, Experiments, Writing
tags: [deep-learning, PyTorch]
math: false                  # set to true if you use LaTeX
---

# 3) Push — GitHub Actions builds and deploys automatically (~3-5 min)
git add _posts/2026-05-27-*.md
git commit -m "post: ..."
git push
```

Each post ends with a short *Stuck points* section: what I got wrong today and how I corrected it.

Full SOP: see the Obsidian workflow doc `40-Projects/论文过程博客 - 工作流手册.md` (private).

## Stack

[Chirpy Jekyll theme](https://github.com/cotes2020/jekyll-theme-chirpy), bootstrapped from its [starter](https://github.com/cotes2020/chirpy-starter). GitHub Actions builds and deploys to GitHub Pages.

## License

Content under CC BY 4.0. Underlying Chirpy starter scaffolding under MIT — see `LICENSE`.
