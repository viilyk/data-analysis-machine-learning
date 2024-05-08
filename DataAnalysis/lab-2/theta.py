from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.tools.validation import array_like

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.tsatools import add_trend


class MyThetaModel:

    def __init__(self, series: pd.Series, period: int, difference: bool = True) -> None:
        self.series_values = series.iloc[:, 0]
        self.period = period
        self.difference = difference
        self.is_seasonal = True

    def test_seasonality(self):
        a = self.series_values
        if self.difference:
            a = np.diff(a)
        rho = acf(a, nlags=self.period, fft=True)
        nobs = a.shape[0]
        stat = nobs * rho[-1] ** 2 / np.sum(rho[:-1] ** 2)
        self.is_seasonal = stat > 2.5

    def del_seasonality(self) -> tuple[np.ndarray, np.ndarray]:
        a = array_like(self.series_values, "series", ndim=1)
        if not self.is_seasonal:
            return a, np.empty(0)
        res = seasonal_decompose(a, model='mul', period=self.period)
        return a / res.seasonal, res.seasonal

    def fit(self) -> MyThetaForecast:
        self.test_seasonality()
        a, seasonal = self.del_seasonality()
        trend = add_trend(a)
        _, s = np.linalg.lstsq(trend, a, rcond=None)[0]
        res = ExponentialSmoothing(a).fit()
        return MyThetaForecast(self, seasonal, s, res.forecast(1))


class MyThetaForecast:

    def __init__(self, model: MyThetaModel, seasonal: np.ndarray, b0: float, single: float) -> None:
        self.model = model
        self.seasonal = seasonal
        self.b0 = b0
        self.single = single

    def forecast(self, steps: int, theta: float = 2) -> pd.Series:
        ses = self.single * np.ones(steps)
        trend = self.b0 * np.ones(steps)
        index = getattr(self.model.series_values, "index", None)
        next_obs = pd.date_range(index[-1], freq=index.freq, periods=2)[1]
        index = pd.date_range(next_obs, freq=index.freq, periods=steps)
        components = pd.DataFrame({"trend": trend, "ses": ses, "seasonal": self.seasonal[:steps]}, index=index)
        return ((theta - 1) / theta * components.trend + np.asarray(components.ses)) * np.asarray(components.seasonal)
