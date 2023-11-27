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

OBS: quanto mais amostras a rede receber, mais bem treinada ela vai ficar, ou seja, a accuracy vai ser maior, ou sejakkkk, o resultado vai ser mais exato
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


'''
EXPLICAÇÃO BRABA
'''

'''
Isso é uma pequena rede neural criada e treinada para realizar a 
tarefa classificação de filmes em "bom" ou "ruim".
'''

'''
No início são gerados 200 avaliações (fakes kk) que são representadas por numeros de 1 a 9.
Se a avaliação for de 7 ou maior, ela é rotulada como "boa", caso menor, é rotulada como "ruim".

Para a ementa da A3, esta sendo utilizado a Algebra Linear (as linhas de código abaixo).
Na camada de entrada, cada avaliação de filme é tratada como um vetor, uma matriz com uma única linha apenas.
A álbegra linear é aplicada para manipular e realizar operações matriciais com esses vetores.
Na camada oculta, é realizado combinações dos pesos de cada neurônio. É basicamente uma multiplicação de
matrix (pesos) por vetor (entradas).
Na camada de saída, é realizado também combinações da camada oculta. Novamente, a multiplicação de matriz e vetor é aplicada, que é uma operação de Algebra Linear.

model = keras.Sequential([
    layers.Input(shape=(1,)), - camada de entrada
    layers.Dense(4, activation='relu'), - camada oculta
    layers.Dense(1, activation='sigmoid') - camada de saída
])
'''

'''
Após, é realizado o treinamento do modelo da rede neural. Ali, a rede ajusta todos os pesos nas camadas.
Ela faz isso usando o algoritmo de descida de gradiente, que é feito com cálculos de derivadas e operações 
de algebra linear.

Por fim, a rede faz previsões com novos dados recebidos, realizando as mesmas operações de Álgebra Linear 
para calcular as ativações das camadas. E finalmente, tem o resultado final de cada avaliação de filme.

:D
'''

'''
EXPLICAÇÃO APROFUNDADA DA ALGEBRA LINEAR COM MATRIZ E VETOR APLICADA NO CÓDIGO

A primeira camada 'layers.Input(shape=(1,))' é a camada de entrada da rede neural. Ela espera um vetor de entrada de tamanho 1.
A segunda camada 'layers.Dense(4, activation='relu')': é a camada oculta da rede neural. Ela possui 4 neurônios e utiliza a função de ativação ReLU.
A terceira e última camada 'layers.Dense(1, activation='sigmoid')' é a camada de saída da rede neural. Ela possui 1 neurônio e utiliza a função de ativação sigmóide.

A álgebra linear se da pels calculos matemáticos que os neurônios executam. Eles fazem operações de multiplicações de pesos pelas entradas e viés.
Essas operações são consideradas calculos de álgebra linear
Por exemplo, nas camadas Dense, cada neurônio está conectado aos neurônios da camada anterior através de um conjunto de pesos, formando uma matriz.
Durante a forward propagation e na backpropagation, operação de multiplicação de matrizes e vetores são feitas para calcular as saídas de rede 
e ajustar os pesos dos neurônios.
Todos os calculos de matrizes e vetores são necessários para preparar os neurônios para o treinamento com os dados de teste, para evitar perdas de
eficiência e possíveis erros.

'''

'''
MULTIPLICAÇÃO DE MATRIZ E VETOR - APLICAÇÃO DA ÁLGEBRA LINEAR PARA A EMENTA DA A3

Na camada oculta com ativação ReLU '(layers.Dense(4, activation='relu'))', a multiplicação de matriz e vetor funciona dessa forma:

- vamos supor que a entrada seja um vetor de tamanho 1 (que é o que ta no código no 'layers.Input(shape=(1,))');
- a camada oculta possui 4 neurônios, cada um com seu conjunto de pesos e um viés associado (a segunda camada).
- os pesos de uma camada densa são representados como uma matriz, onde cada linha representa os pesos de um neurônio.
- se tivermos uma entrada de dimensão (1,) e 4 neurônios na camada oculta, teremos uma matriz de pesos com dimensão (1, 4).
- a forward propagation, a entrada é multiplicada pelos pesos dessa camada e, em seguida, somados os viéses.

Essa operação é uma multiplicação de matriz por vetor, onde o vetor de entrada é multiplicado pela matriz da camada densa.

Quando a rede neural inicia, os viéses, assim como os pesos, são inicializados com valores aleatórios pequenos. 
Ao longo do treinamento, assim como os pesos, os viéses são ajustados iterativamente pelo algoritmo de otimização para 
minimizar a diferença entre as previsões do modelo e as labels verdadeiras dos dados de treinamento.
O viés é necessário para melhorar a capacidade da rede neural de aprender e generalizar a partir dos dados, 
permitindo um ajuste mais flexível da função de ativação e dos limites de decisão, e seu valor é ajustado durante o treinamento da rede.

Se fossemos fazer a estrutura de cálculo na mão, ia ficar da seguinte forma:

- entrada = np.array([2])- o dado de entrada
- pesos_camada_oculta = np.random.rand(1, 4) - o peso da camada oculta com 4 neurônios
- vieses_camada_oculta = np.random.rand(4) - viés da camada oculta, um viés para cada neurônio

Aqui, "entrada" é um vetor de dimensão 1. O "pesos_camada_oculta" representa os pesos da camada oculta, e "vieses_camada_oculta" são os vieses associados a cada neurônio.
O calculo de matriz por vetor (entrada pelo peso) + o viés fica da seguinte forma:

- saida_camada_oculta = np.dot(entrada, pesos_camada_oculta) + vieses_camada_oculta - multiplicação de matriz por vetor (entrada pelos pesos) e adição dos vieses
- saida_camada_oculta_relu = np.maximum(0, saida_camada_oculta) - aplicação da função de ativação ReLU
- print("Saída da camada oculta (ReLU):", saida_camada_oculta_relu)

Nesse código, utilizamos o numpy para facilitar nossos cálculos. A função np.dot() é utilizada para realizar 
a multiplicação de matriz por vetor, e a função np.maximum() é utilizada para aplicar a função de ativação ReLU aos resultados da multiplicação.

'''

'''
FUNÇÃO - ATIVAÇÃO RELU - EMENTA DA A3

Quando definimos nossas camadas, nos adicionados um parâmetro adicional para aprimorar ainda mais a eficiência da rede.
A função de ativação ReLU (Rectified Linear Unit) é uma função usada especialmente em camadas intermediárias (camadas ocultas). 
Ela é simples, mas extremamente eficaz e resolve alguns problemas das funções de ativação mais antigas, como a função sigmoide (que também ta sendo usada no código).

A função ReLU se da pela função:
- f(x)=max(0,x)

Ou seja, para qualquer valor de entrada X, a função ReLU retorna X se X for maior que 0. Em resumo, ela "zera" todos 
os valores negativos e mantém os valores positivos como estão. Nessa rede neural, o X na função ReLU é 
definido pelo resultado da multiplicação da matriz pelo vetor e a adição do viés.

Mas isso tem um problema, como ela zera alguns neurônios, isso os torna inativos, os chamados "neurônios mortos", porque caso um valor de um neurônio seja negativo, a
função ReLU vai zerar. Isso impede que esse neurônio se torne inativo na hora do treinamento. Então, ao mesmo tempo que ela simplifica e estabiliza o treinamento da rede, ela 
também torna a rede um pouco menos produtiva.

'''