# 论文路上 · thesis-blog

Tianhao Wu(吴天昊)的硕士论文准备 + 写作过程博客。

**线上地址(部署完成后):** https://tianhao1314.github.io/thesis-blog/

研究方向:**可扩展的、面向人机协同(任务驱动)的学习式点云 / 3D Gaussian Splatting 压缩**。
导师:[Xiumei Li](https://www.lms.tf.fau.de/faudir/xiumei-li/), FAU Erlangen–Nürnberg。

## 这是什么

一个**学习过程日记**,不是研究成果展示页。每天看了什么论文、学了什么概念、跑了什么实验、卡在哪里、又是怎么爬出来的——都在这里。研究成果展示在主站 [tianhao1314.github.io](https://tianhao1314.github.io)。

## 写一篇新博客

```bash
# 1) 在 _posts/ 创建文件,文件名约定:
#    YYYY-MM-DD-slug.md
#    如 2026-05-27-day2-pytorch-hello.md

# 2) front matter 模板:
---
title: "标题"
date: 2026-05-27 20:30:00 +0200
categories: [学习笔记, 阶段一-地基]
tags: [深度学习, PyTorch]
math: false      # 公式多的话改 true
---

# 3) push 即上线(GitHub Actions 自动构建,~2 分钟)
git add _posts/2026-05-27-*.md
git commit -m "post: ..."
git push
```

详细 SOP 见 Obsidian `40-Projects/论文过程博客 - 工作流手册.md`。

## 技术栈

[Chirpy Jekyll theme](https://github.com/cotes2020/jekyll-theme-chirpy)(基于其 [starter](https://github.com/cotes2020/chirpy-starter)),GitHub Actions 自动构建部署。

## License

博客内容 CC BY 4.0。底层 Chirpy starter 框架 MIT,见 `LICENSE`。
