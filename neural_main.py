import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import random

'''
OBS: presta atenção nisso aqui. Se você quiser que todos os números gerados pelo numpy sejam diferentes a cada
execução do código, coloca esse np.random.seed(0) como comentário.
A seed 0 é a semente fixa do numpy, ou seja, isso garante a reproducibilidade, todos os números gerados serão os mesmos sempre.

pq eu to escrevendo isso? pq eu também sempre esqueço do seed 0kkkk

OBS: quanto mais amostras a rede receber, mais bem treinada ela vai ficar, ou seja, a accuracy vai ser maior, o resultado vai ser mais exato
'''
#np.random.seed(0)
amostras = 5000
filme_reviews = np.random.randint(1, 10, amostras)
label = (filme_reviews >= 7).astype(int)

filme_treino, filme_valida, label_treino, label_valida = train_test_split(filme_reviews, label, test_size=0.2, random_state=42)

modelo = keras.Sequential([
    layers.Input(shape=(1,)),
    layers.Dense(4, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

treinamento = modelo.fit(filme_treino, label_treino, epochs=100, validation_data=(filme_valida, label_valida), verbose=0)

valida_perda, valida_acuracy = modelo.evaluate(filme_valida, label_valida)
acuracy_qlt = "perfeita" if valida_acuracy == 1 else "alta" if valida_acuracy > 0.8 else "média" if valida_acuracy > 0.5 else "ruim"
print(f'Acurácia de validação: {valida_acuracy}, Nível de qualidade: {acuracy_qlt}')

amostra_final = 30
novas_reviews = np.random.randint(1, 15, amostra_final)
predicao = modelo.predict(novas_reviews)

for i in range(len(novas_reviews)):
    avaliacao = "bom" if predicao[i][0] > 0.5 else "ruim"
    print(f"Avaliação: {novas_reviews[i]}, Classificação: {avaliacao}")
