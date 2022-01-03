import sys
import numpy as np
from abess import *


np.random.seed(2)
n = 100
p = 20
k = 3
family = "gaussian"
rho = 0.1

data = make_glm_data(family=family, n=n, p=p, k=k, rho=rho)

def assert_reg(coef):
    if (sys.version_info[0] < 3 or sys.version_info[1] < 6):
        return
    nonzero = np.nonzero(coef)[0]
    new_x = data.x[:, nonzero]
    reg = LinearRegression()
    reg.fit(new_x, data.y.reshape(-1))
    assert_value(coef[nonzero], reg.coef_)

# null
model1 = abessLm()
model1.fit(data.x, data.y)

# predict
y = model1.predict(data.x)

# score
score = model1.score(data.x, data.y)
