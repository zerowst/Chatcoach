# **Towards Communicative Medical Coaching via Latent Chain-of-thought**
Code for paper "Towards Communicative Medical Coaching via Latent Chain-of-thought"

## About the paper

## Generated ChatCoach dataset
### Human annotated dataset
Human anotated dataset consists of 291 conversations containing 1315 doctor statements with corresponding patient response and generated coaching sentences.

The human annotated dataset is now avaiable: [Download annotated dataset](https://github.com/zerowst/Chatcoach/blob/main/Testing/dataset/testing.json)

### The whole set 
The training set consists of 2509 conversations, and the validation set consists of 700 conversations.

The training set is available in Training/coach_train.csv

The validation set is available in Validation/coach_valid.csv

The original finetuning data is available in Training/finetuning.csv.

## Configuration
Model: gpt3.5

## Requirement
* [OpenAI API KEY](https://platform.openai.com/account/api-keys)

* Install all required python dependencies:

python>=3.7

numpy>=1.23.2

openai>=0.23.0

transformers>=4.21.1





