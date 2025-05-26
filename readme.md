- The original repository (NAFNet) [Link](https://github.com/megvii-research/NAFNet.git)
- Modified by SAIT
  - Skip connection (U-Net) removed, utilize cascaded architecture
  - channel expanded (for larger RoI)
  - quantization applied ([PAMS](https://arxiv.org/abs/2011.04212))

- running script
  <pre><code class="language-bash">
    ./scripts/run/train3.sh
    ./scripts/test/test3.sh
  </code></pre>
