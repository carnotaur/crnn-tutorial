Введение в работу __Convolutional Recurrent Neural Networks (CRNN)__ 
Туториал можно просмотреть в jupyter notebook
![Alt text](imgs/crnn-archit.png?raw=true "Original CRNN Architecure")  
Процесс тренировки сети написан в __CRNN Training.ipynb__   
Вместо VGG сети которая использовалсь в статье для основы был использован Resnet18, но ничто не мешает вам поменять ее.  
 
__Requirements__
pytorch >= 1.0.0

Слайды сделаны на основе статей: https://arxiv.org/abs/1507.05717 и https://arxiv.org/abs/1811.04256  

Туториалы по работе _Convolutional Neural Networks_:  
 - http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html  
 - https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148 

Туториалы по _Recurкent Neural Networks (RNN)_:  
 - https://medium.com/explore-artificial-intelligence/an-introduction-to-recurrent-neural-networks-72c97bf0912  

Гит оригинальной имплементации статьи самими авторами на lua: https://github.com/bgshih/crnn  