"""
https://oai.azure.com/portal/be5567c3dd4d49eb93f58914cccf3f02/deployment
clausa gpt4
"""

import time
import requests
import config
import string


def parse_sectioned_prompt(s):

    result = {}
    current_header = None

    for line in s.split('\n'):
        line = line.strip()

        if line.startswith('# '):
            # first word without punctuation
            current_header = line[2:].strip().lower().split()[0]
            current_header = current_header.translate(str.maketrans('', '', string.punctuation))
            result[current_header] = ''
        elif current_header is not None:
            result[current_header] += line + '\n'

    return result


def chatgpt(prompt, temperature=0.7, n=1, top_p=1, stop=None, max_tokens=1024,
                  presence_penalty=0, frequency_penalty=0, logit_bias={}):
    messages = [{"role": "user", "content": prompt}]
    payload = {
        "messages": messages,
        "model": config.CHAT_MODEL,
        "temperature": temperature,
        "n": n,
        "top_p": top_p,
        "stop": stop,
        "max_tokens": max_tokens,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "logit_bias": logit_bias
    }
    for attempt in range(5):
        try:
            r = requests.post(f'{config.API_BASE}/v1/chat/completions',
                headers = {
                    "Authorization": f"Bearer {config.API_KEY}",
                    "Content-Type": "application/json"
                },
                json = payload,
                timeout=None
            )
            if r.status_code == 200:
                break
            time.sleep(1)
        except requests.exceptions.RequestException:
            time.sleep(1)
    else:
        raise RuntimeError(f'chatgpt: API request failed after 5 attempts')
    r = r.json()
    return [choice['message']['content'] for choice in r['choices']]


def instructGPT_logprobs(prompt, temperature=0.7):
    payload = {
        "prompt": prompt,
        "model": config.CHAT_MODEL,
        "temperature": temperature,
        "max_tokens": 1,
        "logprobs": 1,
        "echo": True
    }
    for attempt in range(5):
        try:
            r = requests.post(f'{config.API_BASE}/v1/completions',
                headers = {
                    "Authorization": f"Bearer {config.API_KEY}",
                    "Content-Type": "application/json"
                },
                json = payload,
                timeout=None
            )
            if r.status_code == 200:
                break
            time.sleep(2)
        except requests.exceptions.RequestException:
            time.sleep(2)
    else:
        raise RuntimeError('instructGPT_logprobs: API request failed after 5 attempts')
    r = r.json()
    return r['choices']


