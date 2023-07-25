import openai
import numpy as np
# Note: you need to be using OpenAI Python v0.27.0 for the code below to work


### gpt4 account
openai.api_key = ''

### my account
# openai.api_key = ""

response = openai.ChatCompletion.create(
      model="gpt-4-0314",
      messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
            {"role": "user", "content": "Where was it played?"}
        ]
)

print(response)





