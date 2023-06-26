from t_norms_and_t_conorms import *
from generic_membership_functions import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class YieldPredictionSystem():
    """
    Yield prediction system.
    """
    def __init__(self, 
                 data: pd.DataFrame,    # columns: panicle, growth, yield
                 t_norm: str = 'min',
                 t_conorm: str = 'max',) -> None:
        """
        Initialize the yield prediction system.

        Args:
            data (pd.DataFrame): data to be used for training and testing the system
            t_norm (str, optional): t-norm to be used for evaluating rules. Defaults to 'min'.
            t_conorm (str, optional): t-conorm to be used for evaluating rules. Defaults to 'max'.
        """

        # Check t-norm and t-conorm
        assert t_norm in ['min', 'product', 'lukasiewicz']
        assert t_conorm in ['max', 'sum', 'lukasiewicz']

        # Check data
        assert 'panicle' in data.columns and data['panicle'].dtype == 'float64'
        assert 'growth' in data.columns and data['growth'].dtype == 'float64'
        assert 'yield' in data.columns and data['yield'].dtype == 'float64'
        assert data.columns.size == 3

        # Save percentiles for panicle, growth and yield
        self.panicle_datapoints = {
            'min': data['panicle'].min(),
            'pct10': data['panicle'].quantile(0.1),
            'pct25': data['panicle'].quantile(0.25),
            'pct50': data['panicle'].quantile(0.5),
            'pct75': data['panicle'].quantile(0.75),
            'pct90': data['panicle'].quantile(0.9),
            'max': data['panicle'].max()
        }
        self.growth_datapoints = {
            'min': data['growth'].min(),
            'pct10': data['growth'].quantile(0.1),
            'pct25': data['growth'].quantile(0.25),
            'pct50': data['growth'].quantile(0.5),
            'pct75': data['growth'].quantile(0.75),
            'pct90': data['growth'].quantile(0.9),
            'max': data['growth'].max()
        }
        self.yield_datapoints = {
            'min': data['yield'].min(),
            'pct10': data['yield'].quantile(0.1),
            'pct25': data['yield'].quantile(0.25),
            'pct50': data['yield'].quantile(0.5),
            'pct75': data['yield'].quantile(0.75),
            'pct90': data['yield'].quantile(0.9),
            'max': data['yield'].max(),
        }

        # Set t-norm and t-conorm functions
        if t_norm == 'min':
            self.t_norm = t_norm_min
        elif t_norm == 'product':
            self.t_norm = t_norm_product
        elif t_norm == 'lukasiewicz':
            self.t_norm = t_norm_lukasiewicz

        if t_conorm == 'max':
            self.t_conorm = t_conorm_max
        elif t_conorm == 'sum':
            self.t_conorm = t_conorm_sum
        elif t_conorm == 'lukasiewicz':
            self.t_conorm = t_conorm_lukasiewicz
        
        # Initialize the data and membership functions
        self.data = data
        self.set_membership_functions()

        self.rules = None


    # --- Membership functions ---

    def set_membership_functions(self) -> None:
        """
        Set membership functions for panicle, growth and yield. We use the 10th, 25th, 50th, 75th and 90th percentiles
        to define the piecewise membership functions.
        """

        # Membership functions for panicle
        self.mu_VLP = lambda x: left_trapezoidal_membership_function(x, a=self.panicle_datapoints['pct10'], b=self.panicle_datapoints['pct25'])
        self.mu_LP = lambda x: triangular_membership_function(x, a=self.panicle_datapoints['pct10'], b=self.panicle_datapoints['pct25'], c=self.panicle_datapoints['pct50'])
        self.mu_MP = lambda x: triangular_membership_function(x, a=self.panicle_datapoints['pct25'], b=self.panicle_datapoints['pct50'], c=self.panicle_datapoints['pct75'])
        self.mu_HP = lambda x: triangular_membership_function(x, a=self.panicle_datapoints['pct50'], b=self.panicle_datapoints['pct75'], c=self.panicle_datapoints['pct90'])
        self.mu_VHP = lambda x: right_trapezoidal_membership_function(x, a=self.panicle_datapoints['pct75'], b=self.panicle_datapoints['pct90'])

        # Membership functions for growth
        self.mu_VLG = lambda x: left_trapezoidal_membership_function(x, a=self.growth_datapoints['pct10'], b=self.growth_datapoints['pct25'])
        self.mu_LG = lambda x: triangular_membership_function(x, a=self.growth_datapoints['pct10'], b=self.growth_datapoints['pct25'], c=self.growth_datapoints['pct50'])
        self.mu_MG = lambda x: triangular_membership_function(x, a=self.growth_datapoints['pct25'], b=self.growth_datapoints['pct50'], c=self.growth_datapoints['pct75'])
        self.mu_HG = lambda x: triangular_membership_function(x, a=self.growth_datapoints['pct50'], b=self.growth_datapoints['pct75'], c=self.growth_datapoints['pct90'])
        self.mu_VHG = lambda x: right_trapezoidal_membership_function(x, a=self.growth_datapoints['pct75'], b=self.growth_datapoints['pct90'])

        # Membership functions for yield
        self.mu_VLY = lambda x: left_trapezoidal_membership_function(x, a=self.yield_datapoints['pct10'], b=self.yield_datapoints['pct25'])
        self.mu_LY = lambda x: triangular_membership_function(x, a=self.yield_datapoints['pct10'], b=self.yield_datapoints['pct25'], c=self.yield_datapoints['pct50'])
        self.mu_MY = lambda x: triangular_membership_function(x, a=self.yield_datapoints['pct25'], b=self.yield_datapoints['pct50'], c=self.yield_datapoints['pct75'])
        self.mu_HY = lambda x: triangular_membership_function(x, a=self.yield_datapoints['pct50'], b=self.yield_datapoints['pct75'], c=self.yield_datapoints['pct90'])
        self.mu_VHY = lambda x: right_trapezoidal_membership_function(x, a=self.yield_datapoints['pct75'], b=self.yield_datapoints['pct90'])
        

    # --- Fuzzify input ---

    def fuzzify_input(self) -> None:
        """
        Fuzzify panicle and growth.
        """
        # Fuzzify panicle
        self.data['VLP'] = self.data['panicle'].apply(self.mu_VLP)
        self.data['LP'] = self.data['panicle'].apply(self.mu_LP)
        self.data['MP'] = self.data['panicle'].apply(self.mu_MP)
        self.data['HP'] = self.data['panicle'].apply(self.mu_HP)
        self.data['VHP'] = self.data['panicle'].apply(self.mu_VHP)

        # Fuzzify growth
        self.data['VLG'] = self.data['growth'].apply(self.mu_VLG)
        self.data['LG'] = self.data['growth'].apply(self.mu_LG)
        self.data['MG'] = self.data['growth'].apply(self.mu_MG)
        self.data['HG'] = self.data['growth'].apply(self.mu_HG)
        self.data['VHG'] = self.data['growth'].apply(self.mu_VHG)        

    
    # --- Evaluate rules ---
    
    def evaluate_rules(self, rules: list, data: pd.DataFrame) -> None:
        """
        Evaluate two-to-one rules. Rules are given by tuples of the form 
            (operator, panicle_fuzzy_set, growth_fuzzy_set, yield_fuzzy_set)
        and they are expressions of the form:
            IF panicle is <panicle_fuzzy_set> <operator> growth is <growth_fuzzy_set> THEN yield is <yield_fuzzy_set>

        Example: ('AND', "LP", "LG", "LY") means: IF panicle is LP AND growth is LG THEN yield is LY

        Args:
            rules (list): list of rules in the form of tuples (operator, panicle_fuzzy_set, growth_fuzzy_set, yield_fuzzy_set)
            data (pd.DataFrame): data to be used for evaluating the rules

        Raises:
            ValueError: if the operator is not 'AND' or 'OR'
        """
        # Initialize the aggregated membership values for yield categories
        data['pred_VLY'] = 0.0
        data['pred_LY'] = 0.0
        data['pred_MY'] = 0.0
        data['pred_HY'] = 0.0
        data['pred_VHY'] = 0.0

        # Evaluate each rule
        for rule in rules:
            # Get rule parameters
            operator, panicle_fuzzy_set, growth_fuzzy_set, yield_fuzzy_set = rule[0], rule[1], rule[2], rule[3]

            # Evaluate rule
            if operator == 'AND':
                yield_membership = self.t_norm(data[panicle_fuzzy_set], data[growth_fuzzy_set])
                data['pred_' + yield_fuzzy_set] = self.t_conorm(data['pred_' + yield_fuzzy_set], yield_membership)

            elif operator == 'OR':
                yield_membership = self.t_conorm(data[panicle_fuzzy_set], data[growth_fuzzy_set])
                data['pred_' + yield_fuzzy_set] = self.t_conorm(data['pred_' + yield_fuzzy_set], yield_membership)

            else:
                raise ValueError('Invalid operator')
            

    # --- Defuzzify yield ---

    def defuzzify_yield(self) -> None:
        """
        Defuzzify yield using the centroid method.
        """
        assert 'pred_VLY' in self.data.columns
        assert 'pred_LY' in self.data.columns
        assert 'pred_MY' in self.data.columns
        assert 'pred_HY' in self.data.columns
        assert 'pred_VHY' in self.data.columns

        x = np.linspace(self.yield_datapoints['min'], self.yield_datapoints['max'], 1000)
        y_VLY = self.data['pred_VLY'].apply(
            lambda pred: [self.mu_VLY(x_i) if self.mu_VLY(x_i) < pred else pred for x_i in x]
        )
        y_LY = self.data['pred_LY'].apply(
            lambda pred: [self.mu_LY(x_i) if self.mu_LY(x_i) < pred else pred for x_i in x]
        )
        y_MY = self.data['pred_MY'].apply(
            lambda pred: [self.mu_MY(x_i) if self.mu_MY(x_i) < pred else pred for x_i in x]
        )
        y_HY = self.data['pred_HY'].apply(
            lambda pred: [self.mu_HY(x_i) if self.mu_HY(x_i) < pred else pred for x_i in x]
        )
        y_VHY = self.data['pred_VHY'].apply(
            lambda pred: [self.mu_VHY(x_i) if self.mu_VHY(x_i) < pred else pred for x_i in x]
        )

        # For each x_i save the largest y_i
        y = np.array([np.array([np.max([y_VLY[j][i], y_LY[j][i], y_MY[j][i], y_HY[j][i], y_VHY[j][i]]) for i in range(len(x))]) for j in range(len(self.data))])

        # Calculate the centroid
        self.data['predicted_crisp_yield'] = np.sum(x * y, axis=1) / np.sum(y, axis=1)


    # --- Utility functions ---

    # Plot membership functions

    def plot_panicle_membership_functions(self, figsize: tuple = (8, 4), fontsize: int = 10, title=None) -> None:             
        x = np.linspace(self.panicle_datapoints['min'], self.panicle_datapoints['max'], 1000)
        y_VLP = [self.mu_VLP(x_i) for x_i in x]
        y_LP = [self.mu_LP(x_i) for x_i in x]
        y_MP = [self.mu_MP(x_i) for x_i in x]
        y_HP = [self.mu_HP(x_i) for x_i in x]
        y_VHP = [self.mu_VHP(x_i) for x_i in x]

        plt.figure(figsize=figsize)

        plt.plot(x, y_VLP, label='VLP')
        plt.plot(x, y_LP, label='LP')
        plt.plot(x, y_MP, label='MP')
        plt.plot(x, y_HP, label='HP')
        plt.plot(x, y_VHP, label='VHP')

        plt.xlabel('Number of panicles (million/ha)', fontsize=fontsize)
        plt.ylabel('$\mu$', fontsize=fontsize, rotation=0)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        if title is not None:
            plt.title(title, fontsize=fontsize)    
        plt.legend(fontsize=fontsize)

        plt.tight_layout()
        plt.show()
    
    def plot_growth_membership_functions(self, figsize: tuple = (8, 4), fontsize: int = 10, title=None) -> None:
        x = np.linspace(self.growth_datapoints['min'], self.growth_datapoints['max'], 1000)
        y_VLG = [self.mu_VLG(x_i) for x_i in x]
        y_LG = [self.mu_LG(x_i) for x_i in x]
        y_MG = [self.mu_MG(x_i) for x_i in x]
        y_HG = [self.mu_HG(x_i) for x_i in x]
        y_VHG = [self.mu_VHG(x_i) for x_i in x]

        plt.figure(figsize=figsize)

        plt.plot(x, y_VLG, label='VLG')
        plt.plot(x, y_LG, label='LG')
        plt.plot(x, y_MG, label='MG')
        plt.plot(x, y_HG, label='HG')
        plt.plot(x, y_VHG, label='VHG')

        plt.xlabel('Growth period (days)', fontsize=fontsize)
        plt.ylabel('$\mu$', fontsize=fontsize, rotation=0)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        if title is not None:
            plt.title(title, fontsize=fontsize)
        plt.legend(fontsize=fontsize)

        plt.tight_layout()
        plt.show()
    
    def plot_yield_membership_functions(self, figsize: tuple = (8, 4), fontsize: int = 10, title=None) -> None:
        x = np.linspace(self.yield_datapoints['min'], self.yield_datapoints['max'], 1000)
        y_VLY = [self.mu_VLY(x_i) for x_i in x]
        y_LY = [self.mu_LY(x_i) for x_i in x]
        y_MY = [self.mu_MY(x_i) for x_i in x]
        y_HY = [self.mu_HY(x_i) for x_i in x]
        y_VHY = [self.mu_VHY(x_i) for x_i in x]

        plt.figure(figsize=figsize)

        plt.plot(x, y_VLY, label='VLY')
        plt.plot(x, y_LY, label='LY')
        plt.plot(x, y_MY, label='MY')
        plt.plot(x, y_HY, label='HY')
        plt.plot(x, y_VHY, label='VHY')

        plt.xlabel('Crop yield (t/ha)', fontsize=fontsize)
        plt.ylabel('$\mu$', fontsize=fontsize, rotation=0)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        if title is not None:
            plt.title(title, fontsize=fontsize)
        plt.legend(fontsize=fontsize)

        plt.tight_layout()
        plt.show()