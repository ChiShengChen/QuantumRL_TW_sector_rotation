import numpy as np
import pandas as pd
import config

class SectorRotationEnv:
    """
    A custom environment for the sector rotation task.
    It follows a similar interface to OpenAI Gym.
    """
    def __init__(self, features_df, feature_cols, sequence_length):
        super().__init__()
        self.features_df = features_df
        self.feature_cols = feature_cols
        self.sequence_length = sequence_length
        
        self.unique_dates = self.features_df['date'].unique()
        self.num_industries = self.features_df['industry'].nunique()
        self.current_step = 0
        
        # Action space: choose one industry
        self.action_space_n = self.num_industries
        
        # Observation space: (sequence_length, num_industries * num_features)
        # We flatten the features for all industries for a given day into one vector
        self.flat_features_dim = self.num_industries * len(self.feature_cols)
        self.observation_space_shape = (self.sequence_length, self.flat_features_dim)

    def _get_state(self):
        """
        Retrieves the state for the current time step.
        The state is a sequence of the last `sequence_length` days of features.
        """
        if self.current_step < self.sequence_length - 1:
            # Not enough history, return a zero state
            return np.zeros(self.observation_space_shape)

        # Get the date for the current decision point
        current_date = self.unique_dates[self.current_step]
        
        # Get the window of dates for the state
        start_idx = self.current_step - self.sequence_length + 1
        end_idx = self.current_step + 1
        state_dates = self.unique_dates[start_idx:end_idx]
        
        # Get all features for the dates in the window
        state_df = self.features_df[self.features_df['date'].isin(state_dates)]
        
        # Pivot to have industries as columns and features as rows over time
        # For simplicity, we'll just use the features of a single industry for the state
        # A more complex state would include all industries.
        # This is a simplification to match the sequence format (B, T, C).
        # We assume the agent picks one industry and we provide its history.
        
        # Let's find an industry to return state for. This is tricky.
        # A better approach is to have the state represent the whole market.
        # Let's reshape the state to be (T, num_industries * num_features)
        
        # For now, let's stick to a per-industry decision process, even if simplified.
        # The agent will get the state for one industry and decide on an action.
        # The problem is which industry to provide state for.
        
        # Let's redefine the state to be the features for ALL industries over the sequence length
        # state shape: (sequence_length, num_industries, num_features)
        # We can flatten this if needed by the model.
        
        # Let's try to create the state for a single industry first, and let main loop handle iterating industries
        
        # Let's assume the state is for a specific industry, chosen by the main loop.
        # This simplifies the environment, but pushes logic to the training loop.
        
        # Let's try to make the env self-contained. The state will be for ALL industries.
        
        state_data = []
        for date in state_dates:
            daily_features = self.features_df[self.features_df['date'] == date]
            # sort by industry to have a consistent order
            daily_features = daily_features.sort_values('industry')
            state_data.append(daily_features[self.feature_cols].values)

        # state_data is a list of arrays of shape (num_industries, num_features)
        # We stack them to get (sequence_length, num_industries, num_features)
        state = np.stack(state_data, axis=0)
        
        # The QRWKV model expects (B, T, C) - Batch, Time, Channels/Features
        # Our state is (T, num_industries, num_features). We need to map this.
        # We flatten the last two dimensions: (T, num_industries * num_features)
        state_flattened = state.reshape(self.sequence_length, self.flat_features_dim)

        # The model's input_dim should match this. This is handled in main.py
        return state_flattened

    def reset(self):
        """Resets the environment to the beginning of the time series."""
        self.current_step = self.sequence_length -1
        return self._get_state()

    def step(self, action):
        """
        Executes one time step within the environment.
        - action: The index of the industry chosen by the agent.
        """
        # The action is to select a portfolio of TOP_N industries.
        # The simple PPO agent selects one action. Let's assume the agent's task is to pick the *best* single industry.
        # The problem asks for 10 industries. This implies a multi-action or portfolio construction policy.
        # For now, let's simplify: the agent picks one industry per day.
        
        current_date = self.unique_dates[self.current_step]
        
        # Get the ground truth for the *next* day
        next_day_step = self.current_step + 1
        if next_day_step >= len(self.unique_dates):
            # End of dataset
            return self._get_state(), 0, True, {}
            
        next_date = self.unique_dates[next_day_step]
        
        # Get the ranks for the next day
        next_day_data = self.features_df[self.features_df['date'] == next_date]
        
        # The 'target' column in feature_engineering was if it was in top N.
        # We can use that directly.
        
        industries = sorted(self.features_df['industry'].unique())
        chosen_industry_name = industries[action]
        
        # Find the target for the chosen industry on the *current* date, which looks at the next day's rank
        target_row = self.features_df[(self.features_df['date'] == current_date) & (self.features_df['industry'] == chosen_industry_name)]
        
        if target_row.empty:
            reward = -1 # Should not happen
        else:
            is_in_top_n = target_row['target'].values[0]
            reward = 1.0 if is_in_top_n else -0.1 # Reward 1 if correct, small penalty if not
            
        # Move to the next day
        self.current_step += 1
        
        done = self.current_step >= len(self.unique_dates) -1
        
        next_state = self._get_state()
        
        return next_state, reward, done, {} 