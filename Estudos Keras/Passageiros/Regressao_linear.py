import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

sns.lineplot(x=xtreino, y=ytreino, label='treino')
sns.lineplot(x=xteste, y=yteste, label='teste')
plt.show()

# Modelo1 -  Regressao Linear
regressor = Sequential()
regressor.add(
    Dense(
        1,
        input_dim=1,
        kernel_initializer='Ones',
        activation='linear',
        use_bias=False)
        )
regressor.compile(loss='mean_squared_error', optimizer='adam')

regressor.summary()
regressor.fit(xtreino, ytreino)

y_predict = regressor.predict(xtreino)

sns.lineplot(x=xtreino, y=ytreino, label='treino')
sns.lineplot(x=xtreino, y=y_predict[:, 0], label='ajuste_treino')
plt.show()

# Voltando os dados para numero de passageiros
# Parte do treinamento
resultados = pd.DataFrame(
    data={'tempo': xtreino, 'passageiros': y_predict[:, 0]}
    )
resultado_transf = sc.inverse_transform(resultados)
resultado_transf = pd.DataFrame(resultado_transf)
resultado_transf.columns = ['tempo', 'passageiros']

# Parte da previsao
y_predict_teste = regressor.predict(xteste)
resultados_tese = pd.DataFrame(
    data={'tempo': xteste, 'passageiros': y_predict_teste[:, 0]}
    )
resultado_transf_teste = sc.inverse_transform(resultados_tese)
resultado_transf_teste = pd.DataFrame(resultado_transf_teste)
resultado_transf_teste.columns = ['tempo', 'passageiros']

# Plot dos dados (Dados completos, Treinado e previsto)
sns.lineplot(
    x='tempo',
    y='passageiros',
    data=passageiros,
    label='dado_completo')
sns.lineplot(
    x='tempo',
    y='passageiros',
    data=resultado_transf,
    label='ajuste_treino')
sns.lineplot(
    x='tempo',
    y='passageiros',
    data=resultado_transf_teste,
    label='previsao')
plt.show()


# Modelo2 -  Regressao Linear
regressor2 = Sequential()
regressor2.add(
    Dense(
        8,
        input_dim=1,
        kernel_initializer='random_uniform',
        activation='sigmoid',
        use_bias=False)
        )
regressor2.add(
    Dense(
        8,
        kernel_initializer='random_uniform',
        activation='sigmoid',
        use_bias=False)
        )
regressor2.add(
    Dense(
        1,
        kernel_initializer='random_uniform',
        activation='linear',
        use_bias=False)
        )
regressor2.compile(loss='mean_squared_error', optimizer='adam')

regressor2.summary()
regressor2.fit(xtreino, ytreino, epochs=500)

y_predict = regressor2.predict(xtreino)
y_predict_teste = regressor2.predict(xteste)

sns.lineplot(x=xtreino, y=ytreino, label='treino')
sns.lineplot(x=xteste, y=yteste, label='teste')
sns.lineplot(x=xtreino, y=y_predict[:, 0], label='ajuste_treino')
sns.lineplot(x=xteste, y=y_predict_teste[:, 0], label='previsao')
plt.show()
