<div align="center">

  <h1 align="center">Extrapolated Urban View Synthesis Benchmark</h1>

### [Paper](https://arxiv.org/pdf/2412.05256) | [Project Page](https://ai4ce.github.io/EUVS-Benchmark/) | [Data](https://huggingface.co/datasets/ai4ce/EUVS-Benchmark)
<!-- | [Paper](https://arxiv.org/abs/2402.14650) -->

</div>

## 
<p align="center">
    <!-- <video width="100%" controls>
        <source src="./assets/hospital.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video> -->
    <img src="./assets/Teaser.png" alt="example" width=100%>
    <br>
    <em style="text-align: left;">TLDR: We build a comprehensive real-world benchmark for quantitatively and qualitatively evaluating extrapolated novel view synthesis in large-scale urban scenes.</em>
</p>



## 📖 Abstract

Photorealistic simulators are essential for the training and evaluation of vision-centric autonomous vehicles (AVs). At their core is Novel View Synthesis (NVS), a crucial capability that generates diverse unseen viewpoints to accommodate the broad and continuous pose distribution of AVs. Recent advances in radiance fields, such as 3D Gaussian Splatting, achieve photorealistic rendering at real-time speeds and have been widely used in modeling large-scale driving scenes. However, their performance is commonly evaluated using an interpolated setup with highly correlated training and test views. In contrast, extrapolation, where test views largely deviate from training views, remains underexplored, limiting progress in generalizable simulation technology. To address this gap, we leverage publicly available AV datasets with multiple traversals, multiple vehicles, and multiple cameras to build the first <strong>E</strong>xtrapolated <strong>U</strong>rban <strong>V</strong>iew <strong>S</strong>ynthesis (EUVS) benchmark. Meanwhile, we conduct quantitative and qualitative evaluations of state-of-the-art Gaussian Splatting methods across different difficulty levels. Our results show that Gaussian Splatting is prone to overfitting to training views. Besides, incorporating diffusion priors and improving geometry cannot fundamentally improve NVS under large view changes, highlighting the need for more robust approaches and large-scale training. We have released our data to help advance self-driving and urban robotics simulation technology.

## 🔊News
- 2024/12/9: Our paper is now available on [arXiv](https://arxiv.org/pdf/2412.05256)!
- 2024/12/10: Our data is now available on [Hugging Face](https://huggingface.co/datasets/ai4ce/EUVS-Benchmark)!

## 🗓️ TODO
- [✔] Data release
- [ ] Code release (Releasing very soon)


## 📊 Baseline Code
Here are the official code links for the baseline. 

- [3DGS](https://github.com/graphdeco-inria/gaussian-splatting)
- [3DGM](https://github.com/NVlabs/3DGM)
- [GaussianPro](https://github.com/kcheng1021/GaussianPro)
- [VEGS](https://github.com/deepshwang/vegs)
- [PGSR](https://github.com/zju3dv/PGSR)
- [2DGS](https://github.com/hbb1/2d-gaussian-splatting)
- [feature 3DGS](https://github.com/ShijieZhou-UCLA/feature-3dgs)
- [Zip-NeRF](https://github.com/SuLvXiangXin/zipnerf-pytorch)
- [instant-NGP](https://github.com/NVlabs/instant-ngp)


## 🖊️ Citation
If you find this project useful in your research, please consider cite:

```BibTeX

@misc{han2024extrapolatedurbanviewsynthesis,
      title={Extrapolated Urban View Synthesis Benchmark}, 
      author={Xiangyu Han and Zhen Jia and Boyi Li and Yan Wang and Boris Ivanovic and Yurong You and Lingjie Liu and Yue Wang and Marco Pavone and Chen Feng and Yiming Li},
      year={2024},
      eprint={2412.05256},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.05256}, 
}
```
