import os
import json
import random
from typing import List, Optional

import openai
from dotenv import load_dotenv
load_dotenv()
openai.organization = os.environ.get("OPENAI_ORGANIZATION")
openai.api_key = os.environ.get("OPENAI_API_KEY")

QUALIFIERS = ['very', None, 'a bit']
BIG5 = {
    'EXT': [
        ('unfriendly', 'friendly'),
        ('introverted', 'extraverted'),
        ('silent', 'talkative'),
        ('timid', 'bold'),
        ('unassertive', 'assertive'),
        ('inactive', 'active'),
        ('unenergetic', 'energetic'),
        ('unadventurous', 'adventurous and daring'),
        ('gloomy', 'cheerful')
    ],
    'AGR': [
        ('distrustful', 'trustful'),
        ('immoral', 'moral'),
        ('dishonest', 'honest'),
        ('unkind', 'kind'),
        ('stingy', 'generous'),
        ('unaltruistic', 'altruistic'),
        ('uncooperative', 'cooperative'),
        ('self-important', 'humble'),
        ('unsympathetic', 'sympathetic'),
        ('selfish', 'unselfish'),
        ('disagreeable', 'agreeable')
    ],
    'CON': [
        ('unsure', 'self-efficacious'),
        ('messy', 'orderly'),
        ('irresponsible', 'responsible'),
        ('lazy', 'hardworking'),
        ('undisciplined', 'self-disciplined'),
        ('impractical', 'practical'),
        ('extravagant', 'thrifty'),
        ('disorganized', 'organized'),
        ('negligent', 'conscientious'),
        ('careless', 'thorough')
    ],
    'NEU': [
        ('relaxed', 'tense'),
        ('at ease', 'nervous'),
        ('easygoing', 'anxious'),
        ('calm', 'angry'),
        ('patient', 'irritable'),
        ('happy', 'depressed'),
        ('unselfconscious', 'self-conscious'),
        ('level-headed', 'impulsive'),
        ('contented', 'discontented'),
        ('emotionally stable', 'emotionally unstable')
    ],
    'OPE': [
        ('unimaginative', 'imaginative'),
        ('uncreative', 'creative'),
        ('artistically unappreciative', 'artistically appreciative'),
        ('unaesthetic', 'aesthetic'),
        ('unreflective', 'reflective'),
        ('emotionally closed', 'emotionally aware'),
        ('uninquisitive', 'curious'),
        ('predictable', 'spontaneous'),
        ('unintelligent', 'intelligent'),
        ('unanalytical', 'analytical'),
        ('unsophisticated', 'sophisticated'),
        ('socially conservative', 'socially progressive')
    ],
}


# Big-Five personality profile.
class Big5Profile:
    def __init__(self, ope=None, con=None, ext=None, agr=None, neu=None,
                 n_adj: int = 3, modifiers: Optional[List[Optional[str]]] = None, 
                 shuffle_adj: bool = True
                 ):

        self.modifiers = modifiers if modifiers is not None \
                            else QUALIFIERS                      # ['very', None, 'a bit']

        self.values = list(range(1, len(self.modifiers)+1)) + \
            [(-1) * i for i in range(1, len(self.modifiers)+1)]  # [1, 2, 3, -1, -2, -3]

        self.profile = {
            'OPE': ope or random.choice(self.values),
            'CON': con or random.choice(self.values),
            'EXT': ext or random.choice(self.values),
            'AGR': agr or random.choice(self.values),
            'NEU': neu or random.choice(self.values),
        }
        assert all(val in self.values for val in self.profile.values())

        # randomly select n_adj adjectives for each personality dimension.
        self.persona_text = self._get_persona(n_adj=n_adj, shuffle=shuffle_adj)

    def _get_persona(self, n_adj=3, shuffle=True):
        all_adjs = []
        for domain in self.profile:
            all_adjs += self.get_persona_adjs(domain, n_adj=n_adj)

        # shuffle adjective orders.
        if shuffle:
            random.shuffle(all_adjs)

        return ', '.join(all_adjs)

    def get_persona_adjs(self, domain, n_adj=3):
        assert domain in self.profile
        val = self.profile[domain]

        adjs = [pair[int(val > 0)] for pair in BIG5[domain]]
        if n_adj:
            adjs = random.sample(adjs, n_adj)

        adv = self.modifiers[abs(val) - 1]
        if adv is not None:
            adjs = [f'{adv} {adj}' for adj in adjs]

        return adjs

    def get_profile(self):
        #return [(domain, val > 0, self.modifiers[abs(val) - 1]) 
        #                    for domain, val in self.profile.items()]
        return {domain: val for domain, val in self.profile.items()}


# buyer & seller negotiation agent.
class ConversationLLM(object):
    """
    Usecase of the ConversationLLM class:
    (initilize object)
    llm_buyer = ConversationLLM(model='gpt-4-0613', system_setting=buyer_setting,
                    initial_context=[{"role": "assistant", "content": opening_seller_uttr},
                                     {"role": "user", "content": opening_buyer_uttr}
                                     ])

    (get reponse from current dialogue history, and add to the history.)
    res = llm_buyer.call(add_response_to_context=True)

    (add entry(s) to current dialogue history, often the dialogue opponent's response.)
    llm_buyer.add_to_context(seller_res)

    """

    def __init__(self, model='gpt-4-0613', llm_server=None,
                 system_setting=None, initial_context=None):
        self.model = model
        self.llm_server = llm_server
        self.context = []

        if system_setting:
            self.context.append({
                "role": "system",
                "content": system_setting
            })

        if initial_context:
            self.context += initial_context

    def add_to_context(self, new_input, role='user'):
        self.context.append({
            "role": role,
            "content": new_input,
        })

    def call(self, add_response_to_context=True):
        if self.llm_server is not None:
            response = self._call_llm_server()
        else:
            response = self._call_openai()

        # update context.
        if add_response_to_context:
            self.add_to_context(response, role='assistant')
        return response

    def _call_llm_server(self):
        response = self.llm_server.chat.completions.create(
            model=self.model,
            messages=self.context)

        response = response.choices[0].message.content.strip()
        return response

    def _call_openai(self):
        # get response.
        response = openai.chat.completions.create(
            model=self.model,
            messages=self.context,
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        response = response.choices[0].message.content.strip()
        return response

# for stete detector.
class GPTFunctionCall:
    def __init__(self, vectorize_schema, vectorize_system_instruction=None, model='gpt-4-0613'):
        self.vectorize_schema = vectorize_schema
        self.system_instruction = vectorize_system_instruction

        assert model.startswith('gpt')
        self.model = model

    def call(self, prompt):
        messages = []
        if self.system_instruction:
            messages.append(
                {"role": "system", "content": self.system_instruction})
        messages.append({"role": "user", "content": prompt})

        # query the openai model.
        completion = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            functions=[
                {
                    "name": "get_vector",
                    "parameters": self.vectorize_schema
                }
            ],
            function_call={"name": "get_vector"},
        )

        # convert to dictionary.
        reply_content = completion.choices[0].message
        reply_dict = reply_content.to_dict()['function_call']['arguments']
        
        # note: sometimes there are erros in json-conversion.
        reply_dict = json.loads(reply_dict)
        return reply_dict
