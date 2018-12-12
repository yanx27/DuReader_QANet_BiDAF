# Machine Reading Comprehension on DuReader 

Using QANet and BiDAF on [DuReader](https://github.com/baidu/DuReader). Writen by YanXu, FangYueran and ZhangTianyang<br>
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
* Reference by [BiDAF](https://github.com/allenai/bi-att-flow) and [QANet](https://github.com/NLPLearn/QANet)


