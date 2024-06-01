# ChatCoach Dataset

## Dataset for "Benchmarking Large Language Models on Communicative Medical Coaching: A Dataset and a Novel System"

### Introduction

The ChatCoach dataset encompasses 3,500 dialogues derived from real-world medical consultations found within the MedDialog dataset. This dataset is organized into three subsets for different phases of model development: training, validation, and testing. Specifically, it includes 2,509 conversations for training, 700 for validation, and 291 for testing. The testing set is extensively annotated and acts as a benchmark for evaluating the effectiveness of Large Language Models (LLMs) in medical coaching scenarios.

### Data Structure and Access

- **Training Set:** 
    - **Location:** `Training/coach_train.csv`
    - **Description:** Contains the main body of conversations intended for fine-tuning LLMs.
    - **Additional Resource:** `Training/finetuning.csv` for post-processed instruction-tuning data specifically tailored for Llama2.
    - **Composition:** Each of samples contain input and target.
    
        **Input:** The input prompt contains prior prompt, medical context, dialogue history and doctor's statement. For example,
    
        **prior prompt**: 
        ```Your role is to act as a linguistic coach for a doctor, ensuring their medical advice aligns with the provided context. If discrepancies are identified in the doctor's dialogue compared to the provided medical context, guide them towards making more accurate statements.
        You need to perform the following actions with the specific order:
        1.You need detect the <medical words> in the <doctor's statement>. 
        2.Then you need to make comparison between <medical words> with <medical context>.
        3.If all of <medical words> in <doctor's statement> are relevant to <medical context>, then just respond something encouraging.
        4.If the doctor makes errors regarding symptom descriptions, respond with:
        "Doctor, there seems to be a discrepancy regarding symptom descriptions. The term '<incorrect symptom words>' isn't appropriate for '<current disease>'. Perhaps you should consider '<correct symptoms>'."
        5.For errors related to disease or medication names, use this structure:
        "Doctor, there's an inconsistency with the medical term. '<incorrect words>' isn't right. The appropriate term would be '<correct words>'."
        
        If sentence contains multiple errors, you need to point all of them out based on above patterns.
        
        Please remember that '<incorrect symptom words>' and '<incorrect words>' refer to the misused words you detected. '<current disease>' and '<correct symptoms>' are derived from the medical context provided. 
        Please replace these placeholders with correct medical terms from <medical context> based on above instruction.
        Your response also need to based on <dialogue history> to make your response more suitable in dialogue, but only perform above actions on <doctor's statement>
        Please show me Chinese response directly.
        
        Provided medical context: 
        ```
        
        **medical context**:
        ```
        <medical context>: 肾结石   腰痛    腹痛    腰痛    呕吐    恶心    尿血    下腹痛    腰痛    小便痛    尿频排尿   尿潴留   耻骨上疼痛   肾结石，也称为肾结石（来自拉丁语 r\xc4\x93n\xc4\x93s， 肾脏 和结石， 鹅卵石 ）是一种固体结石或晶体聚集体，由肾脏中的膳食矿物质形成尿。   尿液分析   放射成像程序   血液学测试（血液测试）   X 射线计算机断层扫描（扫描 ct）   全血细胞计数 (Cbc)   静脉补液   肾功能测试（肾功能测试）     酮咯酸（Toradol） ， 氢吗啡酮（Dilaudid） ， 坦索罗辛（Flomax） ， 柠檬酸钾 ， 七氟烷 ， 一氧化二氮 ， 罗库溴铵 ， 氯噻酮 ， 颠茄鸦片 ，  西洛多辛（Rapaflo） ， 麻黄碱（Kie）  
      ```
        **dialogue history**:
        ```
        <dialogue history>:医生：你好，请把你的肾功能测试结果发过来看看。尿结石最大的危害是引起梗阻。所以，直观说还是建议尽早解除梗阻。这个得根据你的结石位置和大小决定手术方式。 
        病人：医生，你好！这是我的CT检查结果。
      ```
        **doctor's statement**:
        ```
        <Doctor's statement>:医生：你好，看了你的CT检查，仍然有肾积水，但是没有看见右侧输尿管有结石存在，建议进一步进行MRU检查，以排除输尿管狭窄和肿瘤的可能。 
      ```
      
      **Target:** coach response
      ```
      教练：医生，关于检查结果的描述可能存在误解。你提到患者有肾积水，但是没有看到右侧输尿管中的结石。建议你进一步提到MRU检查来排除输尿管狭窄和肿瘤的可能性。
      ```
    
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

  The prompts of different methods have been stored in ```multi_prompt/prompts.py```.
        
  To generate coach sentences from different prompting methods, run:
  ```bash
  python multi_prompt/pipeline/run.py

- **Evaluating:**
    
  We have stored some processed lingual coach results of different methods including our method GCoT, gpt3.5,
    Vanilla CoT, etc.
    
  To evaluate the existing methods, run:
  ```bash
  python multi_prompt/pipeline/lingual/evaluating.py
  
* **Note:** 
    
    Before run any of scripts, please add your OPENAI KEY first.
    We used the gpt-3.5-turbo-1106 model for both generating and evaluating our results. 
    Since different versions of the GPT model may yield varying outcomes, 
    this could lead to deviations in the evaluation results.

### Citation
If you find the resource useful, please use the below BibTeX entry to cite the paper:
```
@article{huang2024benchmarking,
  title={Benchmarking Large Language Models on Communicative Medical Coaching: a Novel System and Dataset},
  author={Hengguan Huang, Songtao Wang, Hongfu Liu, Hao Wang, Ye Wang},
  journal={arXiv preprint arXiv:2402.05547},
  year={2024}
}
```




