# Machine Reading Comprehension on DuReader 

Using [BiDAF](https://github.com/allenai/bi-att-flow) and [QANet](https://github.com/NLPLearn/QANet) on [DuReader](https://github.com/baidu/DuReader). Writen by YanXu, FangYueran and ZhangTianyang<br>
### Pretrained embedding
When we train the QANet model, we use the pretrained word embedding from [Baidu Encyclopedia](
https://pan.baidu.com/s/1Rn7LtTH0n7SHyHPfjRHbkg), you can down load and save in folder ./embedding<br>
### Full experimental results
Complete experimental results (including data sets, log of experimental records, tensorboard, and predicted output) can be downloaded from the Baidu network disk：https://pan.baidu.com/s/1qoxnF00wyJ2dqcAPDYTb8w code：gn5b, You can override it with the ./data <br>
<br>
You can also train and test in completed DuReader dataset http://ai.baidu.com/broad/download <br>
# Usage

### BiDAF<br>
Generate dict and embedding：`python BaiduRun.py --prepare`<br>
Train： `python BaiduRun.py --train `<br>
Evaluate on dev： `python BaiduRun.py --evaluate`<br>
Output the answers： `python BaiduRun.py --predict`<br>
![](https://img-blog.csdn.net/20181015145727446?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMyMTEzMTg5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
### QANet<br>
Generate dict and embedding：`python OurRun.py --prepare`<br>
Train： `python OurRun.py --train `<br>
Evaluate on dev： `python OurRun.py --evaluate`<br>
Output the answers：` python OurRun.py --predict`<br>
![](https://img-blog.csdn.net/20180815201736410?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NDk5MTMw/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



