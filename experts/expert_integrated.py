import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from news_database_interface import interface
import yfinance as yf

def get_expert_predictions(beautiful_soup_thing) -> int:
    """
    Get expert prediction from Trendlyne. Returns None if no rating found
    Simply takes the first rating it finds.
    """
    date = None
    for list in beautiful_soup_thing:
        for inner_list in list:
            return_val = None
            for word in inner_list:
                if word in ['Buy', 'buy', 'Bullish', 'bullish', 'Strong Buy', 'strong buy', 'Outperform', 'outperform']:
                    return_val = 1
                elif word in ['Hold', 'hold', 'Neutral', 'neutral', 'Market Perform', 'market perform']:
                    return_val = 0
                elif word in ['Sell', 'sell', 'Bearish', 'bearish', 'Strong Sell', 'strong sell', 'Underperform', 'underperform']:
                    return_val = -1
                if return_val:
                    for word_1 in inner_list:
                        # If the word is a date () then assign that to the date variable
                        if any(x in word_1 for x in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']):
                            date = word_1
                    return return_val, date
    return None, None

def get_sentiment_predictions(ticker: str) -> list[float]:
    """Gets the scores of the several aspects in the ABSA Done by Pratyush and Sirjan"""
    data = interface.get_values_by_ticker(ticker)
    # Convert to numpy array and return
    return np.array(data.values())

def get_stock_price(ticker: str) -> float:
    """
    Get the current stock price for a given ticker.
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        float: Current stock price
    """
    return yf.Ticker(ticker).history(period='1d')['Close'].iloc[0]

def get_stock_growth(ticker: str, date=None) -> float:
    """
    Get the growth of a stock from the previous day.
    
    Args:
        ticker (str): Stock ticker 

    Returns:
        float: Growth of the stock compared to the previous day
    """
    if date:
        stock = yf.Ticker(ticker)
        stock_history = stock.history(period='1d', start=date)
        return (stock_history['Close'].iloc[-1] - stock_history['Close'].iloc[-2]) / stock_history['Close'].iloc[-2]
    
    stock = yf.Ticker(ticker)
    stock_history = stock.history(period='1d', start='2021-01-01')
    return (stock_history['Close'].iloc[-1] - stock_history['Close'].iloc[-2]) / stock_history['Close'].iloc[-2]


class MultiplicativeWeightsExpert:
    def __init__(self, stock: str, experts_list: list[str], learning_rate: float = 0.1, movement_threshold: float = 0.01, verdict_threshold=0.05):
        """
        Initialize the multiplicative weights expert model.
        
        Args:
            stock (str): Stock ticker
            experts_list (list[str]): list of expert names
            learning_rate (float): Learning rate for weight updates
            movement_threshold (float): Threshold for stock movement
        """
        self.stock = stock
        self.experts = experts_list
        self.num_experts = len(experts_list)
        self.learning_rate = learning_rate
        self.movement_threshold = movement_threshold
        self.verdict_threshold = verdict_threshold
        
        # Initialize weights uniformly
        self.weights = np.ones(self.num_experts) / self.num_experts
        
        # Track losses for each expert
        self.losses = np.zeros(self.num_experts)    
        self.cumulative_losses = np.zeros(self.num_experts)
        self.total_loss = 0
        self.weights_history = np.zeros((0, self.num_experts))
        self.expert_losses_history = np.zeros((0, self.num_experts))
        self.total_loss_history = []

        # Track predictions
        self.current_expert_predictions = np.zeros(self.num_experts)
        self.model_rating = 0
        self.verdict = 0    # Buy = 1, Hold = 0, Sell = -1
        
    def get_prediction(self, expert_predictions: list[float]) -> float:
        """
        Get weighted average of expert predictions.
        
        Args:
            expert_predictions (list[float]): list of predictions from each expert
            
        Returns:
            float: Weighted average prediction
        """
        if len(expert_predictions) != self.num_experts:
            raise ValueError("Number of predictions must match number of experts")
               
        return np.dot(self.weights, expert_predictions)
    
    def get_verdict(self, rating: float) -> int:
        if rating > self.verdict_threshold:
            return 1
        if rating < -self.verdict_threshold:
            return -1
        return 0
    
    def calculate_expert_losses(self, expert_predictions: list[float], actual_movement: int) -> np.ndarray:
        """
        Calculate losses for each expert based on their predictions.
        
        Args:
            expert_predictions (list[float]): list of predictions from each expert
            actual_movement (int): 1 if stock increased, -1 if decreased
            
        Returns:
            np.ndarray: Array of losses for each expert
        """
        if actual_movement > self.movement_threshold:
            return np.array([(1 - pred)**2 for pred in expert_predictions])
        elif actual_movement < -self.movement_threshold:
            return np.array([(1 + pred)**2 for pred in expert_predictions])
        else:
            return np.array([(pred)**2 for pred in expert_predictions])
    
    def update_weights(self, expert_losses: np.ndarray):
        """
        Update weights based on expert losses.
        
        Args:
            expert_losses (np.ndarray): Array of losses for each expert
        """
        # Update weights using exponential punishment
        self.weights = self.weights * np.exp(-self.learning_rate * expert_losses)
        
        # Normalize weights
        self.weights = self.weights / np.sum(self.weights)
        
        # Update cumulative losses
        self.cumulative_losses += expert_losses
        self.total_loss += np.dot(self.weights, expert_losses)

        # Save history
        self.weights_history = np.vstack((self.weights_history, self.weights))

    def compute_movement(self, x: float, clipper = lambda x: 1/(1 + np.exp(-x/0.2))) -> int:
        """
        Compute movement based on a sigmoid function.
        
        Args:
            clipper (lambda): Sigmoid function
            x (float): Input value
            
        Returns:
            int: 1 if stock increased, -1 if decreased, smoothed by the clipper
        """
        return clipper(x)
    
    def forward(self, movement: int, clipper=lambda x: 1/(1 + np.exp(-x/0.2))) -> float:
        """
        Forward pass of the model. Computes loss, updates weights and history
        Args: movement (change in stock value (ratio) between -1 to 1)
        Returns: prediction of the model
        """
        # Note, this runs every night

        modified_movement = self.compute_movement(clipper, movement)
        expert_predictions = self.current_expert_predictions
        expert_losses = self.calculate_expert_losses(expert_predictions, modified_movement)
        self.update_weights(expert_losses)

        # Save history
        self.expert_losses_history = np.vstack((self.expert_losses_history, expert_losses))

        # Update the predictions, for the next day
        self.current_expert_predictions = get_sentiment_predictions(self.stock) + [get_expert_predictions(self.stock)]
        self.model_rating = self.get_prediction(self.current_expert_predictions)
        self.verdict = self.get_verdict(self.model_rating)

        return self.model_rating

    def plot_predictions_on_pricing(self, num_days: int):
        """Plots the pricing of the stock, for the last num_days days
        model_predictions are indicates by a greendot for sell, reddot for buy, and yellowdot for hold"""

        #get verdict history from the database
        verdicts = [0 for i in range(num_days)]

        #get the stock pricing history from yfinance
        stock = yf.Ticker(self.stock)
        stock_history = stock.history(period='1d', start='2021-01-01')
        stock_history = stock_history[-num_days:]

        #plot the stock pricing
        plt.plot(stock_history['Close'], label='Stock Price')

        #plot the model predictions
        for i in range(num_days):
            if verdicts[i] == 1:
                plt.plot(stock_history.index[i], stock_history['Close'][i], 'ro')
            elif verdicts[i] == -1:
                plt.plot(stock_history.index[i], stock_history['Close'][i], 'go')
            else:
                plt.plot(stock_history.index[i], stock_history['Close'][i], 'yo')

        plt.show()

    def get_parameters(self) -> dict:
        """
        Get current results and statistics.
        
        Returns:
            Dict: Dictionary containing current weights, losses, and other statistics
        """
        return {
            'weights': self.weights,
            # 'losses': self.losses,
            # 'cumulative_losses': self.cumulative_losses,
            'current_predictions': self.current_expert_predictions,
        }

# # There has to be a model per each stock
# for ticker in database:
#     # make model
#     model = MultiplicativeWeightsExpert(ticker)


# To get weights (you will get the weights in the order of your aspects, trendlyne at the end)
# model.weights

# to get model rating
# model.model_rating

# to get the verdict
# model.verdict

# to do a forward pass (I don't think you need this)
# model.forward(movement)

# Movement is the growth = (final_price - inital_price)/(initial_price)

# Class store everything
class recommendations:
    def __init__(self, stocks: list[str], experts_list: list[str], learning_rate: float = 0.1, movement_threshold: float = 0.01, verdict_threshold=0.05):
        self.models = {stock: MultiplicativeWeightsExpert(stock, experts_list, learning_rate, movement_threshold, verdict_threshold) for stock in stocks}
        self.rating_df = interface.NewsDatabase().to_dataframe()[0]
        # Weights
        self.weight_df = interface.NewsDatabase().to_dataframe()[0]

    def forward(self):
        for stock in self.models:
            # Get the price movement from yfinance (price_today - price_yesterday)/price_yesterday
            movement = get_stock_growth(stock)
            self.models[stock].forward(movement)


data = interface.NewsDatabase().to_dataframe()[1]
# The expert list is the first 8 columns of the dataframe
experts = data.columns[:8]                               


weights_df = pd.DataFrame(columns=experts)
for stock in data["stock_symbol"].unique():
    model = MultiplicativeWeightsExpert(stock, experts)
    stock_data = data[data["stock_symbol"] == stock]
    # Drop the useless columns, the only needed columns are published_date, Earnings	Revenue	Margins	Dividend	EBITDA	Debt	Sentiment
    stock_data = stock_data[["published_date", "Earnings", "Revenue", "Margins", "Dividend", "EBITDA", "Debt", "Sentiment"]]
    # Convert the date to datetime
    stock_data["published_date"] = pd.to_datetime(stock_data["published_date"])

    # sort by the date
    stock_data = stock_data.sort_values(by="published_date")

    # Now make a new dataframe with the same columns but unique dats, where the values in the other columns are the average of the values in the original dataframe
    new_data = pd.DataFrame(columns=stock_data.columns)
    new_data["published_date"] = stock_data["published_date"].unique()
    for column in stock_data.columns[1:]:
        new_data[column] = [stock_data[stock_data["published_date"] == date][column].mean() for date in new_data["published_date"]]
    
    # Now we have the new data, we can start training the model
    for i in range(1, len(new_data)):
        movement = get_stock_growth(stock, new_data["published_date"][i-1])
        model.forward(movement)

    # Save the weights as a row in the weights_df
    weights_df.loc[stock] = model.weights