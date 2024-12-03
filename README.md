# ChatCoach Project

## Implementation for "Benchmarking Large Language Models on Communicative Medical Coaching: A Dataset and a Novel System" [Paper:](https://arxiv.org/pdf/2402.05547)

### Introduction

The ChatCoach dataset encompasses 3,500 dialogues derived from real-world medical consultations found within the MedDialog dataset. This dataset is organized into three subsets for different phases of model development: training, validation, and testing. Specifically, it includes 2,509 conversations for training, 700 for validation, and 291 for testing. The testing set is extensively annotated and acts as a benchmark for evaluating the effectiveness of Large Language Models (LLMs) in medical coaching scenarios.

### Data Structure and Access

- **Training Set:** 
    - **Location:** `Training/coach_train.csv`
    - **Description:** Contains the main body of conversations intended for fine-tuning LLMs.
    - **Additional Resource:** `Training/finetuning.csv` for post-processed instruction-tuning data specifically tailored for Llama2.
     
- **Validation Set:** 
    - **Location:** `Validation/coach_valid.csv`
    - **Purpose:** Provides conversations used to avoid overfitting to the training data.

- **Testing Set:** 
    - **Location:** `Testing/dataset/testing.json`
    - **Highlight:** This set is thoroughly annotated and serves as the primary benchmark for assessing the medical coaching performance of LLMs. It is crucial for final model evaluations, reflecting how well a model can perform in real-world medical coaching scenarios.


### Usage
Our code for interface, generation, evaluation has been released in ```\src```. To use the code, please run ```cd src```.

* **Interface:** 

  We developed a simple interface of ChatCoach based on GPT. You can add your own OPENAI KEY in `interface/run.py` at first.

  To test the interface, run: 
  ```bash
  python interface/run.py

- **Generation:**
    
  We used different prompts for generation and evaluation including our method GCoT, vanilla instruction, 
  vanilla CoT, zero-shot CoT. 
  [The prompts are available here](https://github.com/zerowst/Chatcoach/blob/main/src/multi_prompt/prompts.py)
        
  To generate coach sentences from different prompting methods, run:
  ```bash
  python multi_prompt/pipeline/run.py

- **Evaluation:**
    
  We have stored some processed lingual coach results of different methods including our method GCoT, gpt3.5,
    Vanilla CoT, etc. You may load those files in ```multi_prompt/pipeline/lingual/data```
    
  To evaluate the existing methods, run:
  ```bash
  python multi_prompt/pipeline/lingual/evaluating.py
  ```
  We also provide an example of coach feedback generated by various approaches. 
  [Coach feedback example is available here.](https://github.com/zerowst/Chatcoach/blob/main/src/multi_prompt/coach.png)
  
   Refer to the example, vanilla CoT fails to identify errors in medical terminology, possibly due to lacking integration with external knowledge. While thorough, Zero-shot
   CoT generates overly verbose feedback unsuited for real-time application. In contrast, GCoT identifies errors
   effectively and provides concise and well-structured feedback, demonstrating superior integration of external
   medical knowledge for practical real-time coaching
* **Note:** 
    
    Before run any of scripts, please add your OPENAI KEY first.
    We used the gpt-3.5-turbo-1106 model for both generating and evaluating our results. 
    Since different versions of the GPT model may yield varying outcomes, 
    this could lead to deviations in the evaluation results.

### Citation
If you find the resource useful, please use the below BibTeX entry to cite the paper(ACL Findings 2024):
```
@inproceedings{huang2024acl,
  title={Benchmarking Large Language Models on Communicative Medical Coaching: A Dataset and a Novel System},
  author={Hengguan Huang, Songtao Wang, Hongfu Liu, Hao Wang, Ye Wang},
  year={Findings of the Association for Computational Linguistics: ACL 2024}
}


```




