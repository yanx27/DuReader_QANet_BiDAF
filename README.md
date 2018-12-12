# Machine Reading Comprehension on DuReader 

* Using QANet and BiDAF on Chinese machine reading comprehension dataset
* Writen by YanXu, FangYueran and ZhangTianyang
* When we train the QANet model, we use the pretrained word embedding from [Baidu Encyclopedia百度百科](
https://pan.baidu.com/s/1Rn7LtTH0n7SHyHPfjRHbkg), you can down load and save in folder ./embedding
* Complete experimental results (including data sets, log of experimental records, tensorboard, and predicted output) can be downloaded from the Baidu network disk：https://pan.baidu.com/s/1qoxnF00wyJ2dqcAPDYTb8w code：gn5b, You can override it with the ./data 
#
### BiDAF<br>
generate dict and embedding：`python BaiduRun.py --prepare`<br>
train： `python BaiduRun.py --train `<br>
evaluate： `python BaiduRun.py --evaluate`<br>
test： `python BaiduRun.py --predict`<br>

### QANet<br>
generate dict and embedding：`python OurRun.py --prepare`<br>
train： `python OurRun.py --train `<br>
evaluate： `python OurRun.py --evaluate`<br>
test：` python OurRun.py --predict`<br>
#
* Reference by [DuReader](https://github.com/baidu/DuReader) and [QANet](https://github.com/NLPLearn/QANet)


