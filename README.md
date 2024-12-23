# Analysis of Player Tracking Data Extracted from Football Match Feed
## ML workflow
> **Standards followed for creating End-to-End Machine Learning Workflow:** https://ml-ops.org/content/end-to-end-ml-workflow
<br>
<img src="https://user-images.githubusercontent.com/68152189/163807378-e552e98c-b5e3-472a-965d-3a8fe5cb2255.png" width=500/>
<br>

## Modules
1. <a href="">Object Detection </a>
2. <a href="">Jersy Color and Number Detection </a>
3. <a href="src/track.py">Object Tracking Algorithm </a>
4. <a href="src/extractData.py">Extract and Format Tracking Data </a>
5. <a href="">Model for extracting similar scenarios </a>
6. <a href="">Evaluating player's decision making </a>

<br>

## Software testing
**Reference:** https://www.jeremyjordan.me/testing-ml/

![tesing img](https://www.jeremyjordan.me/content/images/size/w1000/2020/08/Group-5-1.jpg)


* The project uses only pre-trained models for now, so there is no need for Model evaluation. However, there is a need to do **Model Testing** which checks for behaviors that we expect our model to follow.
  
* **Unit tests** which operate on atomic pieces the workflow and check for any bugs.
  
* **Integration testing** where individual modules of the software are combined and tested as a workflow. 

<br>

## Deployment of the project
The application will be built using streamlit and dockerised and deployed on cloud. 
> **Standards followed for deployment:** https://neptune.ai/blog/best-practices-docker-for-machine-learning

<br>

## How to run?
* Clone repo
* Create conda environment  ```conda create --name <env name>```
* Activate environment ```conda activate <env name>```
* ```pip install -r requirments.txt``` <br> **or** <br>
*  ```conda create --name <env name> --file requirments.txt``` <br>
* Run the main file: ```python app.py```

