import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize

import pytensor
import pytensor.tensor as pt

import pymc as pm
import arviz as az


def gen_stimulus(train_trials, test_trials, pwin_dict, seed=42):
    phase1 = ['AX', 'BX', 'AY'] * (train_trials // 3)
    phase2 = ['AX', 'BX', 'AY', 'BY'] * (test_trials // 4)
    rng = np.random.default_rng(seed)
    rng.shuffle(phase1)
    rng.shuffle(phase2)
    Stimulus = phase1 + phase2
    Light_A = [1 if s[0] == 'A' else 0 for s in Stimulus]
    Light_B = [1 if s[0] == 'B' else 0 for s in Stimulus]
    Sound_X = [1 if s[1] == 'X' else 0 for s in Stimulus]
    Sound_Y = [1 if s[1] == 'Y' else 0 for s in Stimulus]
    Stim_AX = [1 if s == 'AX' else 0 for s in Stimulus]
    Stim_AY = [1 if s == 'AY' else 0 for s in Stimulus]
    Stim_BX = [1 if s == 'BX' else 0 for s in Stimulus]
    Stim_BY = [1 if s == 'BY' else 0 for s in Stimulus]
    Phase = ['Learning'] * 360 + ['Test'] * 40

    Pwin = [pwin_dict[s] for s in Stimulus]
    Reward = rng.binomial(1, Pwin)
    data = pd.DataFrame({
        'Phase': Phase,
        'Stimulus': Stimulus,
        'Light_A': Light_A,
        'Light_B': Light_B,
        'Sound_X': Sound_X,
        'Sound_Y': Sound_Y,
        'Stim_AX': Stim_AX,
        'Stim_AY': Stim_AY,
        'Stim_BX': Stim_BX,
        'Stim_BY': Stim_BY,
        'Reward': Reward
    })
    return data

class Model:
    """ How to pred association V based on input Stimuli & Reward"""
    def __init__(self, components):
        """components: list of feature names"""
        self.components = components
        self.reset()
    
    def get_features(self, data):
        return data[self.components]
    
    def get_weights(self):
        return self.weights.copy()
    
    def reset(self):
        self.weights = pd.Series(0.0, index=self.components)
    
    def predict_V(self, features, weights = None):
        """weighted sum of features"""
        if weights is None:
            weights = self.weights
        result= (features * weights)
        if isinstance(result, pd.DataFrame):
            return result.sum(axis=1)
        if isinstance(result, pd.Series):
            return result.sum()

class FeaturalModel(Model):
    """V is predicted based on the presence of individual features (e.g., Light_A, Sound_X)"""
    def __init__(self):
        super().__init__(["Light_A", "Light_B", "Sound_X", "Sound_Y"])
    
    # @staticmethod
    # def feature_to_configural(feature_weights):
    #     return pd.DataFrame({
    #         'Stim_AX': feature_weights['Light_A'] + feature_weights['Sound_X'],
    #         'Stim_BX': feature_weights['Light_B'] + feature_weights['Sound_X'],
    #         'Stim_AY': feature_weights['Light_A'] + feature_weights['Sound_Y'],
    #         'Stim_BY': feature_weights['Light_B'] + feature_weights['Sound_Y']
    #     })
    
    # def get_weight_history(self, type = 'featural'):
    #     """Get the history of weights across trials, with option to return configural or featural weights"""
    #     if type == 'featural':
    #         return self.weight_history.copy()
    #     elif type == 'configural':
    #         return self.feature_to_configural(self.weight_history.copy())
    #     raise ValueError("FeaturalModel only have 'featural' and 'configural' weights")

class ConfiguralModel(Model):
    """V is predicted based on the presence of specific feature combinations (e.g., Stim_AX, Stim_AY)"""
    def __init__(self):
        super().__init__(["Stim_AX", "Stim_AY", "Stim_BX", "Stim_BY"])

    # def get_weight_history(self, type = 'configural'):
    #     """Get the history of weights across trials, with option to return configural or featural weights"""
    #     if type == 'configural':
    #         return self.weight_history.copy()
    #     raise ValueError("ConfiguralModel only have 'configural' weights")


class Agent:
    def __init__(self, model):
        self.model = model
        self.predictions = []
        self.errors = []
        self.weight_history = {}
    
    def reset(self):
        self.model.reset()
        self.predictions = []
        self.errors = []
        self.weight_history = {}
    
    def learn(self, features, reward, **kwargs):
        """How to predict V and update weights based on features and reward"""

        # Prediction
        pred_V = self.model.predict_V(features)
        self.predictions.append(pred_V)

        # Delta
        delta = reward - pred_V
        self.errors.append(delta)

        # Update
        self.update_weights(features, reward, delta, **kwargs)
    
    def update_weights(self, features, reward, delta, **kwargs):
        """Frequency model and RW model are different"""
        raise NotImplementedError

    def run_trial(self, data, **kwargs):
        """Get full evolution of weights and predictions across trials"""
        self.reset()
        for i, row in data.iterrows():

            # save weights before update
            self.weight_history[i] = self.model.get_weights() 

            features = self.model.get_features(row)
            reward = row['Reward']
            self.learn(features, reward, **kwargs)
        self.weight_history[len(data)] = self.model.get_weights()

        result = data.copy()
        result['Prediction'] = self.predictions
        result['Error'] = self.errors
        self.weight_history = pd.DataFrame.from_dict(self.weight_history, orient='index')
        return result
    
    def get_weights(self):
        """Get current weights of the model"""
        return self.model.get_weights()
    
    @staticmethod
    def feature_to_configural(feature_weights):
        return pd.DataFrame({
            'Stim_AX': feature_weights['Light_A'] + feature_weights['Sound_X'],
            'Stim_BX': feature_weights['Light_B'] + feature_weights['Sound_X'],
            'Stim_AY': feature_weights['Light_A'] + feature_weights['Sound_Y'],
            'Stim_BY': feature_weights['Light_B'] + feature_weights['Sound_Y']
        })
    
    def get_weight_history(self):
        """Get the history of weights across trials"""
        if isinstance(self.model, FeaturalModel):
            return self.feature_to_configural(self.weight_history.copy())
        elif isinstance(self.model, ConfiguralModel):
            return self.weight_history.copy()

    
    def predict_V(self, data, **kwargs):
        """Predict the association V based on input data"""
        return self.run_trial(data, **kwargs)['Prediction']
    
    def predict_N(self, pred_V, b):
        """Predict the expected number of actions based on predicted V and bias b"""
        return  np.maximum(b + pred_V, 1e-8)  # Ensure lambda is positive for Poisson distribution
    
    def generate_N(self, pred_V, b, random_state=42):
        """Generate the number of actions based on predicted V and bias b, using a Poisson distribution"""
        pred_N = self.predict_N(pred_V, b)
        if isinstance(pred_N, (pd.Series, pd.DataFrame)):
            pred_N = pred_N.values

        rng = np.random.default_rng(random_state)
        return rng.poisson(pred_N)
    
    def loglikelihood(self, data, b, **kwargs):
        """Calculate the log-likelihood of the observed actions given model parameters"""
        result = self.run_trial(data, **kwargs)
        pred_N = self.predict_N(result['Prediction'].values, b)
        ll = poisson.logpmf(result['Action'].values, pred_N)
        return ll.sum()


class LearningAgent(Agent):
    """V is updated based on the Rescorla-Wagner learning rule"""

    def update_weights(self, features, reward, delta, **kwargs):
        """ΔV = α * (Reward - Predicted V) * Feature Presence"""
        alpha = kwargs.get('alpha', 0.1)
        self.model.weights += alpha * features * delta
    
    @staticmethod
    def rw_scan_step(features, reward, weights, alpha):
        """Updating weights, pytensor version"""
        pred_V = (features * weights).sum()
        delta = reward - pred_V
        new_weights = weights + alpha * features * delta
        return new_weights, pred_V, delta
    
    @staticmethod
    def rw_scan(features, rewards, alpha, init_weight):
        """Run trials, pytensor version"""

        # scan function will iteratively apply rw_scan_step to each trial,
        # feature and reward are the input sequences (pre determined by data structure)
        # weights are updated trial by trial, initialized by given init_weight
        # pred_V and delta are also calculated (no need for initial level)
        # alpha is a non-sequence(constant) argument
        # outputs are pred_V across trials 
        (weight, pred_V, delta), _ = pytensor.scan(
            fn=LearningAgent.rw_scan_step,
            sequences=[features, rewards],
            outputs_info=[init_weight, None, None],
            non_sequences=[alpha]
            )
        return pred_V
    
    def predict_V_pt(self, data, **kwargs):
        """Predict V, pytensor version"""

        # prepare input, transfer to pytensor format
        features = pt.as_tensor(self.model.get_features(data).values.astype(np.float64), dtype = 'float64')
        rewards = pt.as_tensor(data['Reward'].values.astype(np.float64), dtype = 'float64')

        # preare parameters, alpha and init_weight
        n_features = len(self.model.components)
        init_weight = pt.zeros(n_features, dtype = 'float64')
        alpha = kwargs.get('alpha', 0.1)
        alpha = pt.as_tensor(alpha, dtype = 'float64')

        # run scan to get pred_V across trials
        pred_V = self.rw_scan(features, rewards, alpha, init_weight)
        return pred_V

class StatisticalAgent(Agent):
    """V is updated based on the frequency of reward given feature presence"""

    def __init__(self, model):
        """Initialize counts and rewards for each feature"""
        super().__init__(model)
        self.counts = pd.Series(0.0, index=self.model.components)
        self.rewards = pd.Series(0.0, index=self.model.components)
    
    def reset(self):
        super().reset()
        self.counts = pd.Series(0.0, index=self.model.components)
        self.rewards = pd.Series(0.0, index=self.model.components)
    
    def update_weights(self, features, reward, delta, **kwargs):
        """Update weights based on feature rewards divided by feature counts"""
        self.counts += features
        self.rewards += features * reward

        self.model.weights = self.rewards / self.counts.clip(lower = 1)

        ## Important: FeaturalFreq model should use mean instead of sum
        if isinstance(self.model, FeaturalModel):
            self.model.weights /= 2

    def predict_V_pt(self, data, **kwargs):
        """Predict V, pytensor version"""
        pred_V = self.predict_V(data, **kwargs)
        return pt.as_tensor(pred_V.values)

if __name__ == "__main__":
    # Generate data
    pwin_dict = {'AX': 0.2, 'BX': 0.6, 'AY': 0.6, 'BY': 1.0}  # Reward probabilities for each stimulus type
    data = gen_stimulus(train_trials=360, test_trials=40, pwin_dict=pwin_dict)  # 400 trials total, with 360 for training and 40 for testing
    lm = LearningAgent(FeaturalModel())  # Ground truth model
    result_lm = lm.run_trial(data, alpha=0.4)  # learning rate: alpha = 0.4
    data['Action'] = lm.generate_N(result_lm['Prediction'], b=3.0)  # baseline: b = 2.0
    data.to_csv('data.csv', index=False)
