import os
import json
import argparse
import numpy as np
import pandas as pd
from math import log
from scipy.stats import spearmanr, pearsonr, chi2_contingency
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

from Agent import BIG5
stratgy_fn = './data/strategy_list.json'
STRATEGIES = json.load(open(stratgy_fn, 'r'))
STRATEGY_MAPPING = {x: category for category, strat_list in STRATEGIES.items()
                    for x in strat_list}

def extract_dialog_from_dir(parent_dir):
    all_dialogs = []
    for root, dirs, files in os.walk(parent_dir):
        for fn in files:
            if not fn.endswith('json'):
                continue

            with open(os.path.join(root, fn), 'r') as F:
                data = json.load(F)
                all_dialogs += data

    print(f'extracted {len(all_dialogs)} from the directory {parent_dir}')
    return all_dialogs

def negotiation_report(data):
    dialog = data['utterances']
    report = {}
    # negotiation variables.
    report['product_category'] = data['product']['category']
    conv_personality_rank = {-3: 1, -2: 2, -1: 3, 1: 4, 2: 5, 3: 6}
    for role in ['seller', 'buyer']:
        for p_dim in BIG5:
            persona = data[role]['persona_type'][p_dim]
            report[f'{role}_{p_dim}'] = conv_personality_rank[persona] 

            report[f'{role}_price'] = data[role]['target_price']
    
    
    # determine whether it's a successful deal.
    success_rounds = [i for i, uttr in enumerate(dialog)
                        if uttr[2].get('state') == 'accept']

    if not success_rounds:
        report.update({
            'id': data['id'], 'success': 0,
            'n_round': len(dialog), 'n_word': sum(len(uttr[1].split()) for uttr in dialog),
        })
    
    else:
        # extract: length of dialog (before reaching success state), final deal price.
        n_round = success_rounds[0] + 1 
        n_word = sum(len(text.split())
                          for (speaker, text, state) in dialog[:n_round])
        # metrics.
        deal_price = int(dialog[n_round - 1][2]['product price'])
        buyer_ideal_price = int(data['buyer']['target_price'])
        seller_ideal_price = int(data['seller']['target_price'])
        utils = _get_utility_scores(buyer_ideal_price, seller_ideal_price, deal_price)

        # concession rate. buyer/seller.
        buyer_proposals = [
            state.get('product price') for (speaker, text, state) in dialog if speaker == 'buyer']
        seller_proposals = [
            state.get('product price') for (speaker, text, state) in dialog if speaker == 'seller']
        cr = _get_concession_rate(
            buyer_proposals, seller_proposals, 
            buyer_ideal_price, seller_ideal_price)

        report.update({
            'id': data['id'], 'success': 1, 'deal_price': deal_price,
            'n_round': n_round, 'n_word': n_word,
            'seller_util': utils['seller'], 'buyer_util': utils['buyer'], 'joint_util': utils['joint'],
            'seller_cr': cr['seller'], 'buyer_cr': cr['buyer']
        })

    return report

def _get_utility_scores(buyer_init_price, seller_init_price, deal_price,
                       heuristic_ratio=0.7):
    """ Calculate the individual & joint utility scores and fairness (diff. of utility). """
    # pre-calculated values for utilty calculation.
    b_low, s_upper = buyer_init_price, seller_init_price
    s_low = b_low + (s_upper - b_low) * (1 - heuristic_ratio)
    b_upper = b_low + (s_upper - b_low) * heuristic_ratio

    # utilities.
    seller_util = (deal_price - s_low) / (s_upper - s_low)
    buyer_util = (b_upper - deal_price) / (b_upper - b_low)
    joint_util = ((deal_price - s_low) * (b_upper - deal_price)
                      ) / (b_upper - s_low) ** 2

    return {'buyer': buyer_util, 'seller': seller_util,
            'joint': joint_util, 'fairness': abs(buyer_util - seller_util)}

def _get_concession_rate(buyer_proposals, seller_proposals, buyer_init_price, seller_init_price,
                        heuristic_ratio=0.7):
    """Calculate avegraged concession rate across all rounds."""
    # Pre-computed values.
    b_low, s_upper = buyer_init_price, seller_init_price
    s_low = b_low + (s_upper - b_low) * (1 - heuristic_ratio)
    b_upper = b_low + (s_upper - b_low) * heuristic_ratio

    effective_proposals = [price_t for price_t in buyer_proposals + seller_proposals
                            if price_t is not None]
    b_low = min(buyer_init_price, min(effective_proposals))
    s_upper = max(seller_init_price, max(effective_proposals))

    # buyer.
    buyer_cr_per_round = [
        log((p_t - b_low) / (b_upper - b_low) + 0.1)
        for p_t in buyer_proposals if p_t is not None
    ] 
    buyer_cr = sum(buyer_cr_per_round) / len(buyer_proposals)

    # seller.
    seller_cr_per_round = [
        log((s_upper - p_t) / (s_upper - s_low) + 0.1)
        for p_t in seller_proposals if p_t is not None
    ]
    seller_cr = sum(seller_cr_per_round) / len(seller_proposals)

    return {'buyer': buyer_cr, 'seller': seller_cr}

def extract_utterance_dataframe(dialogs):
    df = []
    for d in dialogs:
        report = negotiation_report(d)
        for speaker, text, state in d['utterances']:
            # construct row data.
            row_data = {'speaker': speaker} 
            if 'strategy' in state and state['strategy'] in STRATEGY_MAPPING:
                strategy_raw = state['strategy']
                row_data['strategy'] = STRATEGY_MAPPING[strategy_raw] 

            for p_dim in BIG5:
                persona_level_of_dim = d[speaker]['persona_type'][p_dim] 
                row_data[f'{p_dim}+'] = int(persona_level_of_dim > 0)
                row_data[f'{p_dim}-'] = int(persona_level_of_dim < 0)

            if report['success'] == 1:
                row_data['target_util'] = report[f'{speaker}_util']
                row_data['joint_util'] = report['joint_util']

            df.append(row_data)

    return pd.DataFrame(df)

def personality_analysis(dialogs, target_role=None, visualize=False):
    result_dataframe = pd.DataFrame([negotiation_report(d) for d in dialogs])

    # calculate Spearsman's rank correlation between each metric and personality.
    corr = {}
    metrics = [f'{target_role}_util', 'joint_util', f'{target_role}_cr', 'success', 'n_round']
    for dim in BIG5:
        for metric in metrics:
            # Drop rows with NaN values in either of the columns to avoid errors
            filtered_data = result_dataframe[[f'{target_role}_{dim}', metric]].dropna()
            # Calculate Spearman correlation and p-value
            correlation, p_value = spearmanr(filtered_data[f'{target_role}_{dim}'], 
                                             filtered_data[metric])
            corr[(dim, metric)] = (correlation, p_value) 
    print(corr)

    # visualize.
    if visualize:
        columns = list(BIG5.keys())
        rows = metrics
        rows_repr = ['Target Utility', 'Joint Utility',
                    'Concession R.', 'Success', 'Dialog len.']

        # Create a matrix to hold the correlation values
        correlation_matrix = [[0] * len(columns) for _ in rows]
        p_value_matrix = [[1] * len(columns) for _ in rows]

        for (dim, metric), (correlation, p_value) in corr.items():
            i = rows.index(metric)
            j = columns.index(dim) 
            correlation_matrix[i][j] = correlation
            p_value_matrix[i][j] = p_value

        correlation_matrix = np.array(correlation_matrix)
        p_value_matrix = np.array(p_value_matrix)

        # Normalize p-values for color intensity
        norm = Normalize(vmin=-0.1, vmax=0.1, clip=False)

        # Define two color maps
        positive_cmap = LinearSegmentedColormap.from_list(
            "pos", ['#fb8c34', "white"])
        negative_cmap = LinearSegmentedColormap.from_list(
            "neg", ['#34a3fb', "white"])

        # Plotting
        fig, ax = plt.subplots()
        for i in range(len(rows)):
            for j in range(len(columns)):
                corr = correlation_matrix[i, j]
                p_val = p_value_matrix[i, j]
                color = positive_cmap(
                    norm(p_val)) if corr >= 0 else negative_cmap(norm(p_val))
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color))
                text = f'{corr:.3f}' + ('*' if p_val < 0.1 else '') + \
                    ('*' if p_val < 0.05 else '')
                ax.text(j + 0.5, i + 0.5, text, va='center',
                        ha='center', color='black', fontsize=12)

        # Setting ticks and labels
        ax.set_xticks(np.arange(len(columns)) + 0.5)
        ax.set_yticks(np.arange(len(rows)) + 0.5)
        ax.set_xticklabels(columns, fontsize=12)
        ax.set_yticklabels(rows_repr, fontsize=10, rotation=20)
        ax.set_xlim(0, len(columns))
        ax.set_ylim(0, len(rows))
        ax.invert_yaxis()

        plt.title(f'Spearsman\'s correlation of {args.target_role or "both"}')
        plt.tight_layout()
        plt.show()

def engagement_analysis(dialogs):
    result_dataframe = pd.DataFrame([negotiation_report(d) for d in dialogs])
    result_dataframe = result_dataframe.dropna()  # remove the unsuccessful negotiations.

    # Calculate Pearson correlation coefficients between price and negotiation length.
    corr = {}
    for metric in ['n_round', 'n_word']:
        corr[metric] = pearsonr(result_dataframe['seller_price'], 
                                result_dataframe[metric])

    # Print the statistics of each product category.
    PRODUCT_TYPES = ['phone', 'electronics', 'furniture', 'bike', 'housing', 'car']
    stats = {}
    for cat in PRODUCT_TYPES:
        cat_data = result_dataframe[result_dataframe['product_category'] == cat]

        # print the average price, n-round, n-word of the category.
        stats[cat] = {}
        print(f'===== {cat} =====')
        for val in ['seller_price', 'n_round', 'n_word']:
            avg = cat_data[val].mean()
            var = cat_data[val].var()
            stats[cat][val] = avg
            # Display the results
            print(f'{val}: {round(avg, 2)} ± {round(var, 2)}')

    return corr, stats

def linear_regression_analysis(values_1, values_2):
    assert len(values_1) == len(values_2)
    values_1 = np.array(values_1).reshape(-1, 1)
    values_2 = np.array(values_2)

    # Create a column transformer with OneHotEncoder for the categorical variable
    column_transformer = ColumnTransformer(
        [('category_encoder', OneHotEncoder(), [0])],
        remainder='passthrough')

    # Fit transformer to the data
    x_transformed = column_transformer.fit_transform(values_1)

    # Create and fit the model
    model = LinearRegression()
    model.fit(x_transformed, values_2)

    # The coefficients and intercept of the model
    coefficients = model.coef_

    # Assuming 'x_transformed' and 'model' are already defined and the model is trained
    categories = column_transformer.named_transformers_[
        'category_encoder'].categories_[0]
    
    model_parameter = {feature: coeff for feature, coeff in zip(categories, coefficients)}
    return model_parameter

def strategy_analysis(dialogs, target_role=None, visualize=False):
    uttr_df = extract_utterance_dataframe(dialogs)
    if target_role is not None:
        uttr_df = uttr_df[uttr_df['speaker'] == target_role]
    uttr_df = uttr_df.dropna()

    # correlation of strategy and outcome with logistic regression model.
    models = {}
    for metric in ['target_util', 'joint_util']:
        models[metric] = linear_regression_analysis(uttr_df['strategy'], uttr_df[metric])

        # Print each category with its coefficient
        print(f'##### Linear regression model (x=strategy, y={metric}) #####')
        for feature, coeff in models[metric].items():
            print(f"{feature}: {coeff:.3f}")

    # correlation of strategy and personality.
    if visualize:
        if target_role is None:
            model_coeff = models['joint_util']
        else:
            model_coeff = models['target_util']

        # get list of sorted strategy and list of personality traits.
        strategy_types = sorted(model_coeff, key=model_coeff.get, reverse=True)
        sorted_coefficients = [model_coeff[strat] for strat in strategy_types]
        personality_types = [f'{dim}{pos}' for dim in BIG5 for pos in '+-']

        # visualize the heatmap and the bar graph together.
        # set figure size: ax1 is heatmap and ax2 is bar graph.
        if len(strategy_types) >= 10:
            size = (12, 6)
            gridspec = {'width_ratios': [2.1, 0.7], 'wspace': 0.32}
        else:
            size = (12, 4)
            gridspec = {'width_ratios': [2.5, 1], 'wspace': 0.32}
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=size,
                                    sharey=True, gridspec_kw=gridspec)

        # ax1: heatmap of strategy & personality.
        count_table = [[0 for persona in personality_types]
                    for strat in strategy_types]
        
        for i, strat in enumerate(strategy_types):
            for j, persona in enumerate(personality_types):
                data_of_strat = uttr_df[uttr_df['strategy'] == strat] 
                count_table[i][j] = data_of_strat[persona].sum() 

        # Perform the Chi-Square Test
        data = np.array(count_table)
        chi2, p_value, dof, expected = chi2_contingency(data)
        standardized_residuals = (data - expected) / np.sqrt(expected)

        # Plotting
        MAX = 8
        cax = ax1.imshow(standardized_residuals, cmap='RdBu_r',
                        interpolation='nearest', vmin=-MAX, vmax=MAX, aspect=1)

        fig.colorbar(cax, label='Standardized Residual',
                    location='left', shrink=0.95)

        # Adding row and column labels for clarity
        ax1.set_xticks(np.arange(data.shape[1]))
        ax1.set_yticks(np.arange(data.shape[0]))
        ax1.set_xticklabels(personality_types)
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        ax1.set_yticklabels(strategy_types)
        ax1.set_xlabel('Personality Traits')

        """
        # Add values in each cell for better clarity
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                color = 'w' if abs(standardized_residuals[i, j]) > 1 else 'black'
                # ax1.text(j, i, f'{data[i, j]}', ha='center',
                ax1.text(j, i, f'{round(standardized_residuals[i, j], 2)}', ha='center',
                        va='center', color=color)
        """

        # ax2: bar graph.
        bars = ax2.barh(strategy_types, sorted_coefficients, color=[
            '#fbd0b9' if c >= 0 else '#a3cee3' for c in sorted_coefficients])

        # Add text labels in the middle of each bar
        for bar, value in zip(bars, sorted_coefficients):
            ax2.text(bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, f'{value:.3f}',
                    va='center', ha='center', color='black', fontsize=10)

        # Adds a horizontal line at zero
        ax2.axvline(0, color='gray', linewidth=0.7)
        ax2.set_xlabel('Linear Regression Coeff.')

        fig.subplots_adjust(left=0.1)
        #fig.suptitle(title)
        # plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--result_dir')
    parser.add_argument('-t', '--target_role',
                        choices=['seller', 'buyer'])
    parser.add_argument('--personality_analysis',
                        default=False, action='store_true')
    parser.add_argument('--engagement_analysis',
                        default=False, action='store_true')
    parser.add_argument('--strategy_analysis',
                        default=False, action='store_true')
    parser.add_argument('-v', '--visualize', action='store_true')
    args = parser.parse_args()

    # Extract dialog data from folder.
    all_dialogs = extract_dialog_from_dir(args.result_dir)

    # analysis and visualization.
    if args.personality_analysis:
        personality_analysis(all_dialogs, args.target_role, visualize=args.visualize)

    if args.engagement_analysis:
        engagement_analysis(all_dialogs)
    
    if args.strategy_analysis:
        strategy_analysis(all_dialogs, target_role=args.target_role, visualize=args.visualize)