# [AAAI-25] Noisy Label Calibration for Multi-view Classification
<p align="center">
  🎆 Welcome to the repo of NLC! 🎆  
</p>
<p align="center">
  Paper Link: <a href="https://ojs.aaai.org/index.php/AAAI/article/view/35485">Click Here</a>  
</p>
<p align="center">
  If you find this work helpful, please support us with your stars⭐ and cite our paper!    
</p>
<p align="center">
  This repo contains the official PyTorch implementation of NLC!      
</p>

## 🔥 News
• [2025/04/11] 🔥 Our paper has been published by AAAI Conference on Artificial Intelligence 2025!
## 📖 Introduction
### Motivation
Multi-view data often contains noisy labels due to imperfect annotations, which misleads models and reduces classification performance. Existing methods struggle to separate and correct noisy labels during training.
### Method
The proposed Noisy Label Calibration (NLC) method includes:<br>
  &nbsp;&nbsp;1.Cross-view Ranking Learning (CRL) – reduces heterogeneity and enhances consistency across views.<br>
  &nbsp;&nbsp;2.Label Noise Detection (LND) – identifies noisy labels using confidence scores from reliable neighbors.<br>
  &nbsp;&nbsp;3.Label Calibration Learning (LCL) – corrects noisy labels based on neighbor consensus.<br>
  &nbsp;&nbsp;4.Multi-view classification – uses cross-entropy loss with a penalty term to avoid overconfidence.<br>
![image](https://github.com/sstaree/NLC/blob/c7cb1299bf996c19cd716a8654ecb95984efbd8c/image/framework.jpg)
### Results
![image](https://github.com/sstaree/NLC/blob/c7cb1299bf996c19cd716a8654ecb95984efbd8c/image/framework.jpg)
### Dataset
• All dataset can be downloaded at: <br>
https://pan.baidu.com/s/1uLPfx5lMMqMdYtFUav7jcQ?pwd=6666 <br>
passowrd: 6666 
## 💖 Acknowledgements
We would like to present our sincere thanks to the authors for their contributions to the community!
## 📝 Citation
If you find this repo helpful, please consider citing this paper:<br>
```bibtex
@inproceedings{xu2025noisy,
  title={Noisy label calibration for multi-view classification},
  author={Xu, Shilin and Sun, Yuan and Li, Xingfeng and Duan, Siyuan and Ren, Zhenwen and Liu, Zheng and Peng, Dezhong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={20},
  pages={21797--21805},
  year={2025}
}
```
## 📧 Contact
If you have any questions, please feel free to contact the authors:<br>
Shi-Lin Xu: xushilin990@gmail.com.



