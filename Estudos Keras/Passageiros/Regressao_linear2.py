import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Pre-ambulos para plot (Tamanho da figura e tamanho do texto)
mpl.rcParams['figure.figsize'] = (10, 6)
mpl.rcParams['font.size'] = 22

passageiros = pd.read_csv('Passageiros.csv')
passageiros.head()

# Pre-Processamento
sc = StandardScaler()
sc.fit(passageiros)
dado_escalado = sc.transform(passageiros)

# Dividindo treinamento e teste
x = dado_escalado[:, 0]  # Features - Características - Tempo
y = dado_escalado[:, 1]  # Alvo - Número de passageiros


tamanho_treino = int(len(passageiros)*0.9)  # Pegando 90% dos dados para treino
# O resto vamos reservar para teste
tamanho_teste = len(passageiros)-tamanho_treino

xtreino = x[0:tamanho_treino]
ytreino = y[0:tamanho_treino]

xteste = x[tamanho_treino:len(passageiros)]
yteste = y[tamanho_treino:len(passageiros)]


def separa_dados(vetor, n_passos):
    X_novo, y_novo = [], []

    for i in range(n_passos, vetor.shape[0]):
        X_novo.append(list(vetor.loc[i-n_passos:i-1]))
        y_novo.append(vetor.loc[i])
    X_novo, y_novo = np.array(X_novo), np.array(y_novo)
    return X_novo, y_novo


xtreino_novo, ytreino_novo = separa_dados(pd.DataFrame(ytreino)[0], 1)
xteste_novo, yteste_novo = separa_dados(pd.DataFrame(yteste)[0], 1)


# Modelo -  Regressao Linear
regressor = Sequential()
regressor.add(
    Dense(
        8,
        input_dim=1,
        kernel_initializer='ones',
        activation='linear',
        use_bias=False)
        )
regressor.add(
    Dense(
        64,
        kernel_initializer='random_uniform',
        activation='sigmoid',
        use_bias=False)
        )
regressor.add(
    Dense(
        1,
        kernel_initializer='random_uniform',
        activation='linear',
        use_bias=False)
        )
regressor.compile(loss='mean_squared_error', optimizer='adam')

regressor.summary()
regressor.fit(xtreino_novo, ytreino_novo, epochs=100)


y_predict = regressor.predict(xtreino_novo)
y_predict_teste = regressor.predict(xteste_novo)

sns.lineplot(x='tempo', y=ytreino_novo, data=passageiros[1:129], label='treino')
sns.lineplot(x='tempo', y=pd.DataFrame(y_predict)[0], data=passageiros[1:129], label='teste')
sns.lineplot(x='tempo', y=yteste_novo, data=passageiros[130:144], label='teste')
sns.lineplot(x='tempo', y=pd.DataFrame(y_predict_teste)[0].values, data=passageiros[130:144], label='previsao')

plt.show()
