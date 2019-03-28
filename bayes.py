from hyperopt import fmin, tpe, hp, Trials
import time


class SpacePramater:
    def __init__(self, **kwargs):
        pass

    def compute(self, label):
        pass

    def transform(self, index):
        pass


class Choice(SpacePramater):
    def __int__(self, choice_list):
        self.choice_list = choice_list

    def compute(self, label):
        return hp.choice(label, self.choice_list)

    def transform(self, index):
        return self.choice_list[index]


class RandInt(SpacePramater):
    def __int__(self, low: int = 0, high: int = 1):
        self.down = low
        self.up = high

    def compute(self, label):
        return hp.randint(label, self.up - self.down)

    def transform(self, index):
        return self.down + index


class Uniform(SpacePramater):
    def __init__(self, low: float = 0, high: float = 1):
        self.low = low
        self.high = high

    def compute(self, label):
        return hp.uniform(label, self.low, self.high)

    def transform(self, index):
        return index


class QUniform(SpacePramater):
    def __init__(self, low: float = 0, high: float = 1, q: float = 1):
        self.low = low
        self.high = high
        self.q = q

    def compute(self, label):
        return hp.quniform(label, self.low, self.high, q)

    def transform(self, index):
        return index


class LogUniform(SpacePramater):
    def __init__(self, low: float = 0, high: float = 1):
        self.low = low
        self.high = high

    def compute(self, label):
        return hp.loguniform(label, self.low, self.high)

    def transform(self, index):
        return index


class QLogUniform(SpacePramater):
    def __init__(self, low: float = 0, high: float = 1, q: float = 1):
        self.low = low
        self.high = high
        self.q = q

    def compute(self, label):
        return hp.loguniform(label, self.low, self.high, self.q)

    def transform(self, index):
        return index


class Normal(SpacePramater):
    def __init__(self, mu: float = 0, sigma: float = 1):
        self.mu = mu
        self.sigma = sigma

    def compute(self, label):
        return hp.normal(label, self.mu, self.sigma)

    def transform(self, index):
        return index


class LogNormal(SpacePramater):
    def __init__(self, mu: float = 0, sigma: float = 1):
        self.mu = mu
        self.sigma = sigma

    def compute(self, label):
        return hp.lognormal(label, self.mu, self.sigma)

    def transform(self, index):
        return index


class QLogNormal(SpacePramater):
    def __init__(self, mu: float = 0, sigma: float = 1, q: float = 1):
        self.mu = mu
        self.sigma = sigma
        self.q = q

    def compute(self, label):
        return hp.lognormal(label, self.mu, self.sigma, self.q)

    def transform(self, index):
        return index


class BayesGo:
    def __init__(self, space, fn, max_time: int = 600):
        self._space = space
        self.trials = Trials()
        self.current_pos = 0
        self.fn = fn
        self.max_time = max_time
        self._best = None
        self._start_time = None

        self.hp_space = dict()
        self._hp_space()

    def _hp_space(self):
        for k, v in self._space.items():
            self.hp_space[k] = v.compute(k)

    def _fn(self, kwargs):
        param = dict()
        for k, v in kwargs.items():
            param[k] = self._space[k].transform(v)

        return self.fn(**param)

    def search(self, steps=1):
        self.current_pos += steps

        if self.max_time > 0 and self._start_time is None:
            self._start_time = time.time()

        if self.max_time > 0 and time.time() - self._start_time > self.max_time:
            return None

        self._best = fmin(fn=self.fn,
                          space=self.hp_space,
                          algo=tpe.suggest,
                          max_evals=self.current_pos,
                          trials=self.trials)

        if self.max_time > 0 and time.time() - self._start_time > self.max_time:
            return None

    def best_parameter(self):
        """
        推断的最优参数
        :return:
        """
        param = dict()
        for k, v in self._best.items():
            param[k] = self._space[k].transform(v)
        return param

    def max_parameter(self):
        """
        已经搜索过的最优参数
        :return:
        """
        param = dict()
        for k, v in self.trials.best_trial['vals'].items():
            param[k] = self._space[k].transform(v[0])
        return param
