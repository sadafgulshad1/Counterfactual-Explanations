# Interpreting_Adversarial_Examples_with_Attributes
1. Fine tune Resnet-152 for CUB dataset by by running 
```1.CUBFinetune.py```
2. For testing the Finetuned network run ```2.CUB_test.py```
3. Execute the code in **3.Adversarial_attack_code** according to instructions in readme and generate adversarial examples.
4. For testing Finetuned network's performance on adversarial examples run ```4.CUB_test_adv.py```
5. For predicting the attributes for clean as well as adversarial test images run ```5.CUB_SJE.py```. (This code will first train SJE network on clean training images and then predict the attributes for clean and adversarial test images)
6. Run the file ```CUB_adv_train.py``` in the folder named as **6.Adversarial_defense_code** for creating the defense against adversarial examples through adversarial training.  
For testing adversarialy trained network's performance on adversarial examples run ```7.CUB_test_adv_AT.py```.
