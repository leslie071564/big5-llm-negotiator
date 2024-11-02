import os
import json
import argparse
from Game import Bargain


def load_agent_configs(agent_config_fn, n_repeat=1):
    agents_configs = json.load(
        open(agent_config_fn, 'r', encoding='utf-8'))
    n_config = len(agents_configs)

    print(
        f'Loaded {n_config} agent-pair settings from {agent_config_fn}.')

    return agents_configs

def load_llm_server():
    from openai import OpenAI
    llama_server = os.environ.get("LLM_SERVER")
    server = OpenAI(
        base_url=llama_server,
        api_key="EMPTY"
    )
    return server

def main(args):
    # load server.
    server = load_llm_server() if args.use_llm_server else None

    # check output directory.
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # load agent ettings.
    agents_configs = load_agent_configs(args.agent_config_fn)

    for conf in agents_configs:
        data = []
        for i in range(args.n_repeated_trial):
            conf_id = f"{conf['id']}#{i}"

            print(f'===== {conf_id} =====')
            game = Bargain(id=conf_id, config=conf, 
                        model=args.model, llm_server=server)
            dialog = game.run_game(
                max_dialog_len=args.max_dialog_len, kickoff_dialog=True)

            data.append({'id': game.id, 'product': conf['product'], 
                         'seller': conf['seller'], 'buyer': conf['buyer'], 
                         'utterances': dialog})

        # save to file.
        output_fn = os.path.join(args.out_dir, f'{conf_id}.json')

        with open(output_fn, 'w') as file:
            json.dump(data, file, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_config_fn', required=True)
    parser.add_argument('--max_dialog_len', type=int, default=10)
    parser.add_argument('--n_repeated_trial', type=int, default=1)
    parser.add_argument('--use_llm_server', action='store_true')
    parser.add_argument('--model', default="gpt-4-0613")
    parser.add_argument('-o', '--out_dir',
                        default='./data/synthetic_dialogs')
    args = parser.parse_args()

    main(args)
