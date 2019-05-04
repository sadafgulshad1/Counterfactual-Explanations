# Interpreting_Adversarial_Examples_with_Attributes
This repository provides the code for our paper [Interpreting Adversarial Examples with Attributes](https://arxiv.org/pdf/1904.08279.pdf). The abstract of the paper is given as following 
## Abstract
Deep computer vision systems being vulnerable to imperceptible and carefully crafted noise have raised questions regarding the robustness of their decisions. We take a step back and approach this problem from an orthogonal direction. We propose to enable black-box neural networks to justify their reasoning both for clean and for adversarial examples by leveraging attributes, i.e. visually discriminative properties of objects. We rank attributes based on their class relevance, i.e. how the classification decision changes
when the input is visually slightly perturbed, as well as image relevance, i.e. how well the attributes can be localized on both clean and perturbed images. We present comprehensive experiments for attribute prediction, adversarial example generation, adversarially robust learning, and their qualitative and quantitative analysis using predicted attributes on three benchmark datasets.
### Prerequisites
For executing this code you need to install 
* Pytorch0.3
* Python2.6
* Caffe (for grounding model)

1. Fine tune Resnet-152 for CUB dataset by by running 
```1.CUBFinetune.py```
2. For testing the Finetuned network run ```2.CUB_test.py```
3. Execute the code in **3.Adversarial_attack_code** according to instructions in readme and generate adversarial examples.
4. For testing Finetuned network's performance on adversarial examples run ```4.CUB_test_adv.py```
5. For predicting the attributes for clean as well as adversarial test images run ```5.CUB_SJE.py```. (This code will first train SJE network on clean training images and then predict the attributes for clean and adversarial test images)
6. In the folder named as **6.Adversarial_defense_code**.
  i. Run the file ```CUB_adv_train.py``` for creating the defense against adversarial examples through adversarial training. 
  ii. For testing adversarialy trained network's performance on clean test images run ```2.CUB_test.py```.
  iii. For testing adversarialy trained network's performance on adversarial test images run ```3.CUB_test_adv.py```.
  iv. For predicting the attributes for clean as well as adversarial test images on adversarialy trained network run ```4.CUB_SJE.py```. (This code will first train SJE network on clean training images and then predict the attributes for clean and adversarial test images)
  v. For Analysis between attributes predicted for perturbed images when correctly classified with adversarial trainig and missclassified without it run ```5.CUB_Analysis.py```.
7. For testing adversarialy trained network's performance on adversarial examples run ```7.CUB_test_adv_AT.py```.
8. For Analysis between attributes predicted for images correctly classified when clean and missclassified without it run ```8.CUB_Analysis.py```.
9. Execute the code in **3.Adversarial_attack_code** according to instructions in readme and generate examples with random noise. ```run_attack_iter_CUB_random.py```
10. For testing Finetuned network's performance on examples with random noise run ```9.CUB_test_random_noise.py```.
11. Analysis for randomly noised images could be performed in the same way as adversarial examples.
12. For grounding predicted attributes for clean images on clean and adversarial images on adversarial execute code ```demo_copy5.py``` in folder **11.Grounding/bottom-up-attention/tools**. 
## Acknowledgments

* Code from [rwightman/pytorch-nips2017-attack-example](https://github.com/rwightman/pytorch-nips2017-attack-example) is adapted for generating adversarial examples.
* Code from [yqxian/GCPR_Tutorial](https://github.com/sadafgulshad1/GCPR_Tutorial/tree/master/demo/sje) is adapted for predicting attributes.
* Code from [wanglouis49/pytorch-adversarial_box](https://github.com/wanglouis49/pytorch-adversarial_box) is adapted for creating defenses.
* Code from [peteanderson80/bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) is adapted for grounding attributes.
