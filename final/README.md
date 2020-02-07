# Final Project - Domain Adaptation

### Package
pandas==0.25.3  
numpy==1.17.4  
cv2==4.1.2  
PIL==6.2.1  
torch==1.3.1  
torchvision==0.4.2  

### Usage
* download data and model : 
  ```shell
  bash download.sh
  ```
* best model (MCD) : 
  * train : 
    ```shell
    bash MCD_train.sh <trainX> <trainY> <testX>
    ```
  * test : 
    ```shell
    bash MCD_test.sh <testX> <output>
    ```
