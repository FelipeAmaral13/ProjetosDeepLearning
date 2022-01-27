import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.keras.layers import LSTM, Dense

bike = pd.read_csv('bicicletas.csv')
bike['datas'] = pd.to_datetime(bike['datas'])


# Pre-Processamento
sc = StandardScaler()
sc.fit(bike['contagem'].values.reshape(-1, 1))
dado_escalado = sc.transform(bike)

y = sc.transform(bike['contagem'].values.reshape(-1, 1))

tamanho_treino = int(len(bike)*0.9)  # Pegando 90% dos dados para teste
# O resto vamos reservar para teste
tamanho_teste = len(bike)-tamanho_treino

ytreino = y[0:tamanho_treino]
yteste = y[tamanho_treino:len(bike)]


def separa_dados(vetor, n_passos):
    X_novo, y_novo = [], []

    for i in range(n_passos, vetor.shape[0]):
        X_novo.append(list(vetor.loc[i-n_passos:i-1]))
        y_novo.append(vetor.loc[i])
    X_novo, y_novo = np.array(X_novo), np.array(y_novo)
    return X_novo, y_novo


xtreino_novo, ytreino_novo = separa_dados(pd.DataFrame(ytreino)[0], 10)
xteste_novo, yteste_novo = separa_dados(pd.DataFrame(yteste)[0], 10)

# A entrada de redes recorrentes deve possuir a seguinte forma para a entrada
# (número de amostras, número de passos no tempo,
# e número de atributos por passo no tempo).

xtreino_novo = xtreino_novo.reshape(
    (xtreino_novo.shape[0], xtreino_novo.shape[1], 1))
xteste_novo = xteste_novo.reshape(
    (xteste_novo.shape[0], xteste_novo.shape[1], 1))

# Modelo - LSTM
recorrente = Sequential()

recorrente.add(
    LSTM(128, input_shape=(xtreino_novo.shape[1], xtreino_novo.shape[2])))
recorrente.add(Dense(units=1))
recorrente.compile(loss='mean_squared_error', optimizer='adam')

recorrente.summary()
reultado = recorrente.fit(
    xtreino_novo,
    ytreino_novo,
    validation_data=(xtreino_novo, ytreino_novo),
    epochs=100)

y_predict = recorrente.predict(xtreino_novo)
y_predict_teste = recorrente.predict(xteste_novo)

sns.lineplot(
    x='datas', y=ytreino[:, 0], data=bike[0:tamanho_treino], label='treino')
sns.lineplot(
    x='datas', y=y_predict[:, 0], data=bike[0:15662], label='ajuste_treino')
sns.lineplot(
    x='datas', y=yteste[:, 0], data=bike[tamanho_treino:len(bike)],
    label='teste')
sns.lineplot(
    x='datas', y=y_predict_teste[:, 0], data=bike[tamanho_treino+10:len(bike)],
    label='previsao')
plt.xticks(rotation=70)
plt.show()


# Avaliação
plt.plot(reultado.history['loss'])
plt.plot(reultado.history['val_loss'])
plt.show()
