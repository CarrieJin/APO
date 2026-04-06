import re
from abc import ABC, abstractmethod
from typing import List, Dict, Callable
from liquid import Template

import utils
import tasks

class GPT4Predictor(ABC):
    def __init__(self, opt):
        self.opt = opt

    @abstractmethod
    def inference(self, ex, prompt):
        pass

class BinaryPredictor(GPT4Predictor):
    categories = ['No', 'Yes']

    def inference(self, ex, prompt):
        prompt = Template(prompt).render(text=ex['text'])
        response = utils.chatgpt(
            prompt, max_tokens=4, n=1,
            temperature=self.opt['temperature'])[0]
        pred = 1 if response.strip().upper().startswith('YES') else 0
        return pred


class GSM8KPredictor(GPT4Predictor):

    def _extract_answer(self, response):
        """Extract and normalize the numerical answer from the model response.

        Tries \\boxed{...} first, then falls back to the last number in the text.
        Returns a normalized integer string (e.g. '72'), or '' if nothing found.
        """
        # Try \boxed{...}
        match = re.search(r'\\boxed\{([^}]+)\}', response)
        if match:
            content = match.group(1).strip().replace(',', '')
            try:
                val = float(content)
                return str(int(val)) if val == int(val) else str(val)
            except ValueError:
                return content

        # Fallback: last number in the response (handles negative, decimals, commas)
        numbers = re.findall(r'-?\d[\d,]*\.?\d*', response)
        if numbers:
            content = numbers[-1].replace(',', '')
            try:
                val = float(content)
                return str(int(val)) if val == int(val) else str(val)
            except ValueError:
                return content

        return ''

    def inference(self, ex, prompt):
        prompt = Template(prompt).render(text=ex['text'])
        response = utils.chatgpt(
            prompt, max_tokens=512, n=1,
            temperature=self.opt['temperature'])[0]
        return self._extract_answer(response)
