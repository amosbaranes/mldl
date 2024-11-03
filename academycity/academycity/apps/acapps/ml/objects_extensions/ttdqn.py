import os

#
from ..basic_ml_objects import BaseDataProcessing, BasePotentialAlgo
from ....core.utils import log_debug, clear_log_debug
#
from ....core.utils import log_debug, clear_log_debug
#
import pandas as pd
import numpy as np
#
import tensortrade.env.default as default
from tensortrade.data.cdd import CryptoDataDownload
from tensortrade.feed.core import Stream, DataFeed
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import USD, BTC, ETH
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.agents import DQNAgent
#


# we'll create a couple of indicators:
def rsi(price: Stream[float], period: float) -> Stream[float]:
    r = price.diff()
    upside = r.clamp_min(0).abs()
    downside = r.clamp_max(0).abs()
    rs = upside.ewm(alpha=1 / period).mean() / downside.ewm(alpha=1 / period).mean()
    return 100*(1 - (1 + rs) ** -1)


def macd(price: Stream[float], fast: float, slow: float, signal: float) -> (Stream[float], Stream[float]):
    fm = price.ewm(span=fast, adjust=False).mean()
    sm = price.ewm(span=slow, adjust=False).mean()
    md = fm - sm
    signal = md - md.ewm(span=signal, adjust=False).mean()
    return md, signal## Create features with the feed module


# Based on Chapter 11:
# --------------------------
class TDQNAlgo(object):
    def __init__(self, dic):  # to_data_path, target_field
        # print("90567-8-000 Algo\n", dic, '\n', '-'*50)
        try:
            super(TDQNAlgo, self).__init__()
        except Exception as ex:
            print("Error 9057-010 Algo:\n"+str(ex), "\n", '-'*50)

        self.app = dic["app"]


class TDQNDataProcessing(BaseDataProcessing, BasePotentialAlgo, TDQNAlgo):
    def __init__(self, dic):
        # print("90567-010 DataProcessing\n", dic, '\n', '-' * 50)
        super().__init__(dic)
        # print("9005 DataProcessing ", self.app)
        self.PATH = os.path.join(self.TO_OTHER, "tdqn")
        os.makedirs(self.PATH, exist_ok=True)
        # print(f'{self.PATH}')
        self.model = None
        self.lose_list = None
        clear_log_debug()
        #

    def train(self, dic):
        print("4040-40-tdqn: \n", "="*50, "\n", dic, "\n", "="*50)

        # epochs = int(dic["epochs"])
        cdd = CryptoDataDownload()

        data = cdd.fetch("Bitstamp", "USD", "BTC", "1h")
        print(data)

        # choosing the closing price:
        features = []
        for c in data.columns[1:]:
            s = Stream.source(list(data[c]), dtype="float").rename(data[c].name)
            features += [s]

        cp = Stream.select(features, lambda s: s.name == "close")

        # adding the three features (trend indicator, RSI, MACD):
        features = [
            cp.log().diff().rename("lr"),
            rsi(cp, period=20).rename("rsi"),
            macd(cp, fast=10, slow=50, signal=5)[1].rename("macd")
        ]

        feed = DataFeed(features)
        feed.compile()

        for i in range(20):
            print(i, feed.next())

        # setting up broker and the portfolio:
        bitstamp = Exchange("bitstamp", service=execute_order)(
            Stream.source(list(data["close"]), dtype="float").rename("USD-BTC")
        )

        portfolio = Portfolio(USD, [
            Wallet(bitstamp, 10000 * USD),
            Wallet(bitstamp, 10 * BTC)
        ])

        # renderer:
        renderer_feed = DataFeed([
            Stream.source(list(data["date"])).rename("date"),
            Stream.source(list(data["open"]), dtype="float").rename("open"),
            Stream.source(list(data["high"]), dtype="float").rename("high"),
            Stream.source(list(data["low"]), dtype="float").rename("low"),
            Stream.source(list(data["close"]), dtype="float").rename("close"),
            Stream.source(list(data["volume"]), dtype="float").rename("volume")
        ])

        # the trading environment:
        env = default.create(
            portfolio=portfolio,
            action_scheme="managed-risk",
            reward_scheme="risk-adjusted",
            feed=feed,
            renderer_feed=renderer_feed,
            renderer=default.renderers.PlotlyTradingChart(),
            window_size=20
        )

        print("env.observer.feed.next()\n", env.observer.feed.next(), "\n")

        # training a DQN trading agent
        agent = DQNAgent(env)
        print("Y15")

        # this generate an error:
        agent.train(n_steps=200, n_episodes=2, save_path="agents/")

        print("Y16")
        performance = pd.DataFrame.from_dict(env.action_scheme.portfolio.performance, orient='index')
        print(performance)

        result = {"status": "ok tdqn"}
        return result


    # For Multiple Independent variables - World Bank Example
    # We create a special Object to manage this example.

