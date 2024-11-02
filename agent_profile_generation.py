import csv
import json
import argparse
from Agent import Big5Profile

def parse_csv(file_path):
    with open(file_path, mode='r', encoding='utf-8', errors='ignore') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--product_list_fn', default='./data/negotiation_settings.csv')
    parser.add_argument('--n_adj', type=int, default=3)
    parser.add_argument('--n_trial_per_product', type=int, default=2)
    parser.add_argument(
        '--out_fn', default='./data/agent_profiles.json')
    args = parser.parse_args()

    products = parse_csv(args.product_list_fn)
    game_settings = []
    for pd in products:
        product_id, product_name  = pd['id'], pd['short_name']
        if product_name == 'None':
            continue

        for i in range(args.n_trial_per_product):
            game_conf = {'id': f"{product_id}#{i}",
                         'product': {
                            'name': product_name, 
                            'category': pd['category'], 
                            'description': pd['description']} 
                         }

            for role in ['seller', 'buyer']:
                prof = Big5Profile(n_adj=args.n_adj)
                game_conf[role] = {'target_price': pd[f'{role}_price'], 
                                    'persona_type': prof.get_profile(),
                                    'persona_text': prof.persona_text}

            game_settings.append(game_conf)

    # save to file.
    with open(args.out_fn, 'w', encoding='utf-8') as F:
        json.dump(game_settings, F, indent=4)

    print(f"{len(game_settings)} data successfully written to file: {args.out_fn}")
