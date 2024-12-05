<div align="center">

  <h1 align="center">Extrapolated Urban View Synthesis Benchmark</h1>

### [Project Page](https://ai4ce.github.io/EUVS-Benchmark/) | [Data](https://huggingface.co/datasets/ai4ce/EUVS-Benchmark)
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
    <em>TLDR: We make camera localization more generalizable by addressing the data gap via 3DGS and learning gap via a two-branch joint learning with adversarial loss, achieving localization accuracy surpassing 1cm/0.3° in indoor scenarios, 20cm/0.5° in outdoor scenarios, and 10cm/0.2° in driving scenarios.</em>
</p>



## 📖 Abstract

Photorealistic simulators are essential for the training and evaluation of vision-centric autonomous vehicles (AVs). At their core is Novel View Synthesis (NVS), a crucial capability that generates diverse unseen viewpoints to accommodate the broad and continuous pose distribution of AVs. Recent advances in radiance fields, such as 3D Gaussian Splatting, achieve photorealistic rendering at real-time speeds and have been widely used in modeling large-scale driving scenes. However, their performance is commonly evaluated using an interpolated setup with highly correlated training and test views. In contrast, extrapolation, where test views largely deviate from training views, remains underexplored, limiting progress in generalizable simulation technology. To address this gap, we leverage publicly available AV datasets with multiple traversals, multiple vehicles, and multiple cameras to build the first <strong>E</strong>xtrapolated <strong>U</strong>rban <strong>V</strong>iew <strong>S</strong>ynthesis (EUVS) benchmark. Meanwhile, we conduct quantitative and qualitative evaluations of state-of-the-art Gaussian Splatting methods across different difficulty levels. Our results show that Gaussian Splatting is prone to overfitting to training views. Besides, incorporating diffusion priors and improving geometry cannot fundamentally improve NVS under large view changes, highlighting the need for more robust approaches and large-scale training. We have released our data to help advance self-driving and urban robotics simulation technology.


## 🗓️ TODO
- [ ] Data release
- [ ] Code release

## 🖊️ Citation
If you find this project useful in your research, please consider cite:

```BibTeX

@article{han2024euvs,
  title={Extrapolated Urban View Synthesis Benchmark}, 
  author={Xiangyu Han and Zhen Jia and Boyi Li and Yan Wang and Boris Ivanovic and Yurong You and Lingjie Liu and Yue Wang and Marco Pavone and Chen Feng and Yiming Li},
  journal={arXiv preprint arXiv:},
  year={2025},
}
```
