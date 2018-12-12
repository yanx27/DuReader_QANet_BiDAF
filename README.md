# Machine Reading Comprehension on DuReader 

Using [BiDAF](https://github.com/allenai/bi-att-flow) and [QANet](https://github.com/NLPLearn/QANet) on [DuReader](https://github.com/baidu/DuReader). Writen by YanXu, FangYueran and ZhangTianyang<br>
### Pretrained embedding
When we train the QANet model, we use the pretrained word embedding from [Baidu Encyclopedia](
https://pan.baidu.com/s/1Rn7LtTH0n7SHyHPfjRHbkg), you can down load and save in folder ./embedding<br>
### Full experimental results
Complete experimental results (including data sets, log of experimental records, tensorboard, and predicted output) can be downloaded from the Baidu network disk：https://pan.baidu.com/s/1qoxnF00wyJ2dqcAPDYTb8w code：gn5b, You can override it with the ./data <br>
# Usage

### BiDAF<br>
Generate dict and embedding：`python BaiduRun.py --prepare`<br>
Train： `python BaiduRun.py --train `<br>
Evaluate on dev： `python BaiduRun.py --evaluate`<br>
Output the answers： `python BaiduRun.py --predict`<br>

### QANet<br>
Generate dict and embedding：`python OurRun.py --prepare`<br>
Train： `python OurRun.py --train `<br>
Evaluate on dev： `python OurRun.py --evaluate`<br>
Output the answers：` python OurRun.py --predict`<br>
# Reference
[1] Yu, A. W., Dohan, D., Luong, M. T., Zhao, R., Chen, K., Norouzi, M., & Le, Q. V. (2018). QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension. arXiv preprint arXiv:1804.09541.
[2] Seo, M., Kembhavi, A., Farhadi, A., & Hajishirzi, H. (2016). Bidirectional attention flow for machine comprehension. arXiv preprint arXiv:1611.01603.
[3] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in Neural Information Processing Systems (pp. 5998-6008).
[4] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
[5] Xiong, C., Zhong, V., & Socher, R. (2016). Dynamic coattention networks for question answering. arXiv preprint arXiv:1611.01604.
[6] Srivastava, R. K., Greff, K., & Schmidhuber, J. (2015). Highway networks. arXiv preprint arXiv:1505.00387.
[7] Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer normalization. arXiv preprint arXiv:1607.06450.
[8] Chollet, F. (2017). Xception: Deep learning with depthwise separable convolutions. arXiv preprint, 1610-02357.
[9] Weissenborn, D., Wiese, G., & Seiffe, L. (2017). Making neural QA as simple as possible but not simpler. arXiv preprint arXiv:1703.04816.



