import json
import tqdm
from Agent import ConversationLLM, GPTFunctionCall
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


class Bargain:
    END_STATES = ('accept', 'breakdown')

    def __init__(self, id, config,
                 model='gpt-4-0613', llm_server=None
                 ):
        self.id = id
        self.product = config['product']['name']
        self.product_description= config['product']['description']

        self.model = model
        self.llm_server = llm_server
        self.history = []

        # initialize agents.
        self.config = config
        self.seller = self.set_seller(**config['seller']) 
        self.buyer = self.set_buyer(**config['buyer']) 
        self.observer = self.set_state_detector()

    def set_seller(self, target_price, persona_text=None, **kwargs):
        # Seller's instruction text.
        seller_setting = f'Act as a seller that that sells a {self.product}, bargains with the buyer to get a higher deal price. Your reply should not be too long. Your listing price for this item is {target_price}.'

        if self.product_description:
            seller_setting += f' The detail of the product is the following: \n{self.product_description}'

        if persona_text:
            seller_setting += '\n\n' + \
                f'You have following personality: {persona_text}.\n'
            seller_setting += 'Reflect your personality in the negotiation process.'

        # initialize an LLM agent.
        return ConversationLLM(model=self.model, llm_server=self.llm_server, system_setting=seller_setting)

    def set_buyer(self, target_price, persona_text=None, **kwargs):
        # Buyer's instruction text.
        buyer_setting = f'Act as a buyer and try to strike a deal for a {self.product} with a lower price through conversation. Your reply should not be too long. You would like to pay for {target_price}. You can accept higher price though if the item is really good or there are other perks.'
        # add this if we want to add strategy instructions.
        # buyer_text += '\n\n' + \
        #    'During negotiation, actively incorporate negotiation strategies such as appeal to sympathy, embellishments, etc.'

        if persona_text:
            buyer_setting += '\n\n' + \
                f'You have following personality: {persona_text}.\n'
            buyer_setting += 'Reflect your personality in the negotiation process.'

        return ConversationLLM(model=self.model, llm_server=self.llm_server, system_setting=buyer_setting)

    def set_state_detector(self):
        vectorize_schema = {
            "type": "object",
            "properties": {
                "product price": {"type": "number", "format": "integer",
                                  "description": "The average price of the product offered by the last speaker."},
                "state": {"type": "string", "enum": ['offer', 'pondering', 'accept', 'breakdown', 'chit-chat']},
                "strategy": {"type": "string", "description": "Describe the negotiation strategy of the speaker."}
            },
            "required": ["state"]
        }

        vectorize_system_instruction = f"You will be given a partial dialogue in which a buyer and a seller and a buyer negotiate about a deal. Predict the average product price, dialogue state and the strategy of the last speaker by the end of the dialogue."
        observer = GPTFunctionCall(vectorize_schema=vectorize_schema,
                                   vectorize_system_instruction=vectorize_system_instruction, model='gpt-4-0613')
        return observer

    def set_kickoff(self):
        opening_seller_uttr = "Hi, how can I help you?"
        opening_buyer_uttr = f"Hello, I'm interested in your {self.product}. Could you please tell me the price?"

        # buyer context update.
        self.buyer.add_to_context(opening_seller_uttr, role='assistant')
        self.buyer.add_to_context(opening_buyer_uttr, role='user')

        # seller context update.
        self.seller.add_to_context(opening_seller_uttr, role='user')
        self.seller.add_to_context(opening_buyer_uttr, role='assistant')

        # history management.
        self.history += [
            ['seller', opening_seller_uttr, {'state': 'chit-chat'}],
            ['buyer', opening_buyer_uttr, {'state': 'chit-chat'}]
        ]

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(2))
    def run_game(self, max_dialog_len=10, kickoff_dialog=False):
        # kick-off with predefined dialog snippet.
        if kickoff_dialog:
            self.set_kickoff()

        # generate dialogue.
        for _ in range(max_dialog_len):
            # seller.
            seller_res = self.seller.call()
            self.buyer.add_to_context(seller_res)

            # buyer
            buyer_res = self.buyer.call()
            self.seller.add_to_context(buyer_res)

            # get dialogue state and history.
            self._update_history('seller', seller_res)
            self._update_history('buyer', buyer_res)

            # check termination condition.
            if self.if_terminate():
                break

        return self.history

    def _update_history(self, agent_role, response, state=None):
        def _get_observer_prompt(context):
            target_speaker = context[-1][0]
            task_instruction = f"Given a partial dialogue in which a buyer and a seller negotiate about a pro. Predict the average product price, dialogue state and the strategy of the {target_speaker} by the end of the dialogue."

            # dialog context.
            user_input = '[The dialogue]\n'
            for speaker, uttr, *_ in context:
                user_input += f'{speaker}: {uttr}\n'
            user_input = user_input.strip()
            return task_instruction + '\n' + user_input

        self.history.append([agent_role, response, state])
        if state is None:
            prompt = _get_observer_prompt(self.history)
            self.history[-1][2] = self.observer.call(prompt)

    def if_terminate(self):
        # terminate if a deal is make/break.
        return self.history[-1][2]['state'] in Bargain.END_STATES or \
            self.history[-2][2]['state'] in Bargain.END_STATES


class Questionnaire(object):
    def __init__(self, questionnaire_fn, agent_args, model='gpt-4-0613') -> None:
        self.model = model
        self.questions = self.load_questions(questionnaire_fn)
        self.answers = {}

        # initialize agent.
        self.agent = self.set_agent(agent_args)
        self.agent_args = agent_args

    def load_questions(self, questionnaire_fn):
        def _convert_statement(text):
            if not text.startswith('I '):
                text = f'I {text[0].lower()}{text[1:]}'
            if not text.endswith('.'):
                text = text + '.'
            return text

        with open(questionnaire_fn, 'r') as F:
            data = json.load(F)

        for d in data:
            d['statement'] = _convert_statement(d['statement'])

        return data

    def set_agent(self, persona_text):
        setting = f"Act as person with the following personality: {persona_text}."
        vectorize_schema_personality_test = {
            "type": "object",
            "properties": {
                "answer": {"type": "number", "enum": [1, 2, 3, 4, 5]}
            },
            "required": ["answer"]
        }

        self.agent = GPTFunctionCall(vectorize_schema=vectorize_schema_personality_test,
                                     vectorize_system_instruction=setting, model=self.model)
        return self.agent

    def run_game(self):
        for i, question in tqdm.tqdm(enumerate(self.questions)):
            try:
                res = self.run_round(i)
            except:
                res = None

            # extract only the 'answer' key.
            res = res.get('answer', None)

            self.answers[i] = {'statement': question['statement'], "dimension": question['dimension'],
                               "math": question['math'], "answer": res}
        return self.answers

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(2))
    def run_round(self, i):
        statement = self.questions[i]['statement']
        prompt = f'Evaluating the statement, {statement}. '
        prompt += 'Please rate how accurately this describes you on a scale from 1 to 5 (where 1 = "very inaccurate", 2 = "moderately inaccurate", 3 = "neither accurate nor inaccurate", 4 = "moderately accurate", and 5 = "very accurate"). Your answer should be a single digit.'

        res = self.agent.call(prompt=prompt)
        return res
