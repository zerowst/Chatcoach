# ChatCoach Dataset

## Dataset for "Benchmarking Large Language Models on Communicative Medical Coaching: A Dataset and a Novel System"

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




