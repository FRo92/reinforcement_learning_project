# Técnicas de Reinforcement Learning aplicado al juego Snake 🐍 #

Documentación del código utilizado para desarrollar el proyecto final del curso Deep Learning Avanzado @ MIA UC, titulado "Benchmark de metodologías de Aprendizaje reforzado para Snake Game"

## Integrantes: ##
- José Antonio Lipari
- José Francisco Mallea
- Francisca Rojas
- Damaris Saavedra

A continuación se presenta el código usado para realizar el benchmark de las 3 técnicas de aprendizaje reforzado aplicado al juego Snake, cada carpeta alberga una metodología distinta, además se dejaron los archivos csv que contienen la información de las 2000 iteraciones que se compararon y presentaron en el informe final.

## Consideraciones: ##
Se trabajó sobre código desarrollado por otros autores disponibles en la web, referenciados a continuación:
* [Policy Gradient](https://gist.github.com/ViniTheSwan/66fd59d78e94e06e00595ae9c1748d10#file-reinforce-py)
* [deep_q_learning](https://github.com/vedantgoswami/SnakeGameAI)
* [q_learning](https://gist.github.com/jl4r1991)

La ejecución y adopción de los códigos anteriormente señalados no es trivial, puesto que carecen de documentación y algunas funciones se encuentran deprecadas, por lo que gran parte del tiempo se invirtió en entender los códigos y lograr hacerlos funcionar en nuestras máquinas locales.

Se implementó un ambiente virtual seguro donde la ejecución de los tres métodos fuera compatible, los requerimientos de este ambiente se encuentran disponibles en el archivo [requeriments.txt](https://github.com/FRo92/reinforcement_learning_project/blob/main/requeriments.txt) de este repositorio.

Tambiém, se trabajó en la homologación de las condiciones del entorno para que sean comparables (tamaño y forma del tablero, tamaños de serpiente y manzana) dejando un tablero de 600x400 pixeles o bien 30x20 cuadrantes para cada implementación y además homologamos los movimientos de la serpiente.


### ¿Cómo ejecutar el código? 💻 ### (agregar aporte de jugada humana)
Primero se deben instalar las librerías necesarias (se recomienda crear un ambiente seguro) ejecutando el siguiente comando en la terminal (python -v 3.8.10):

```
pip install -r requirements.txt
```
Luego, existen instrucciones para cada implementación para iterar ~2000 veces el juego

## Deep Q Learning 🥇 ##

Es el [método](https://github.com/FRo92/reinforcement_learning_project/tree/main/deep_q_learning) más actual, se basa en el q_learning tradicional, es decir, se busca resolver la ecuación de Bellman, pero se busca representar la función Q con una red neuronal. En este caso se implementó en pytorch una red de una capa oculta de tamaño 256.

Esta implementación requiere tunear los parámetros: learning rate (factor de aprendizaje), penalización (factor de olvido) y epsilon (factor de exploración incial).

```
make deep_q_learning
```

Mientras se ejecuta el juego se desplegará una ventana emergente con la interfaz del juego que permite ver el proceso de aprendizaje, los valores de cada juego quedarán impresos en la terminal.

## Q Learning 🥈 ##

Para el método [dql](https://github.com/FRo92/reinforcement_learning_project/tree/main/q_learning_tradicional) es el más antiguo de la comparación, se basa en la solución de la ecuación de Bellman:

$$
Q(s,a) = R(s,a) + \lambda max_{a'\in A}Q(s',a')
$$

Donde Q(s,a) es el valor que se busca llenar de forma tabular en función de estados s y acciones a, $\lambda$ es el factor de olvido de recompensas lejanas y $max_{a'\in A}Q(s',a')$ es la acción que maximiza la recompensa futura.

Esta implementación requiere ajustar los parámetros $\lambda$ y el factor de exploración $\epsilon$.

Para ejecutar este método, debemos primero debemos inicializar los valores de Q y luego ejecutar el agente con la siguiente instrucción:
```
make q_learning
```
Mientras se ejecuta el juego se desplegará una ventana emergente con la interfaz del juego que permite ver el proceso de aprendizaje, los valores de cada juego quedarán impresos en la terminal.

## Policy Gradient 🥉 ##

El método [policyGradient](https://github.com/FRo92/reinforcement_learning_project/tree/main/policy_gradient), a diferencia de los métodos anteriores, busca aprender la política directamente como una función de probabilidad, en este caso, discreta usando [tfp.distributions.Categorical](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Categorical). Además, la política se actualiza cada vez que la serpiente muere. Se busca maximizar la probabilidad de aquellas acciones que aumentan la recompenza y minimizar aquellas que la disminuyen, lo que se traduce en la siguiente función de pérdida:

$$
\ell(\theta) = -log \pi(a_t | s_t)G_t 
$$

Donde $\pi$ es la política y $G_t$ es la recompensa total.
Aplicando el descenso de gradiente a la función de pérdida se obtiene:

$$
\theta \leftarrow \theta + \eta \nabla log \pi(a_t | s_t)G_t 
$$

Esta implementación está hecha en tenssorflow con dos capas densas de tamaño 30.

Para el método dql ejecutamos la instrucción:
```
make policy_gradient
```
Este método guarda los valores de cada juego en un archivo .csv y va guardando gifs cada vez que se alcanza un record de recomensas, quedan en la carpeta scores y gifs respectivamente.

## Desafíos de preparación y homologación ⚙️ ##
Se investigan 8 implementaciones realizadas por autores diferentes, de las cuales se seleccionaron 3 las que fueron implementadas con las siguientes consideraciones:
* Habilitación hardware propio ya que no ejecutan en colab las implementaciones utilizadas, por problemas de versionamiento o interfaz gráfica.
* GPU utilizada Nvidia: GTX 1660TI  16GB ram en entorno
* CPU utilizada: intel I9                     16GB ram en entono
* Instalación y aprendizaje de librerías:
Pygame /Pytorch /imageio/scipy.ndimage/Tensorflow/matplotlib.pyplot
Implementación de entornos virtuales específicos para cada técnica  ya que las versiones GPU presentaban conflictos de instalación:
* Se utilizó Anaconda instalando versiones específicas de cada implementación.

## Métricas de comparación ⚖️ ##
Para poder realizar una comparación efectiva debemos definir un entorno de comparación justo y modificar las diversas implementaciones para que sean comparables entre ellas, con ese objetivo tomamos las siguientes consideraciones:
Ajustamos el tamaños del tablero a 30x20 cuadrantes o movimientos en todas las implementaciones
Comparamos mejoras en puntaje obtenido vs cantidad de juegos de entrenamiento para definir qué algoritmo aprende más rápido comparando puntaje promedio y máximo.

Incorporamos datos de un humano aprendiendo en condiciones similares como referencia
Adicionalmente, comparamos el tiempo de implementación, ya que realizar un juego demora tiempos distintos dependiendo del algoritmo utilizado.

Finalmente, preparamos versiones distribuídas utilizando CUDA y versiones no distribuidas usando CPU para comparar los beneficios en términos de ahorro de tiempo de entrenamiento, sin embargo, el código presentado en este repositorio solo presenta la implementación en CPU.

## Resultados 📈 ##
A continuación, se presentan los resultados obtenidos al comparar el rendimiento de las tres metodologías:
* Resultados comparación Humano vs Aprendizaje reforzado para 60 iteraciones:
![comparación 60 iteraciones Humano vs Máquina](https://github.com/FRo92/reinforcement_learning_project/blob/main/results_images/resultados_1.png)
Para tener un punto de comparación y referencia entrenamos un [humano](https://www.linkedin.com/search/results/all/?heroEntityKey=urn%3Ali%3Afsd_profile%3AACoAACtV7soBYqUskGfD1tRhBlnIzp756eJ64xg&keywords=jos%C3%A9%20antonio%20lipari%20arias&origin=RICH_QUERY_TYPEAHEAD_HISTORY&position=0&searchId=78036010-c4b8-4767-97f9-a1243137a39a&sid=%409V) sin experiencia previa en el juego y registramos cómo evoluciona su puntaje respecto a los algoritmos de aprendizaje reforzado (2 bloques de 45 minutos de entrenamiento).

Se observa que el humano aprende mucho más rápido que cualquiera de los métodos probados cuando se comparan solo 60 iteraciones, registrando un máximo de 41 puntos  y un promedio de 31 puntos una vez que aprende. De acá surge la pregunta, ¿cuántas iteraciones más que un humano requiere una sistema para llegar o incluso sobrepasar el desempeño humano?

* Resultados comparación Humano vs Aprendizaje reforzado para 1800 iteraciones:
![comparación 1800 iteraciones Humano vs Máquina](https://github.com/FRo92/reinforcement_learning_project/blob/main/results_images/resultados_3.png)

Se puede ver que el mejor desempeño lo obtiene algoritmo deep Q learning después de la iteración 300, seguido de cerca por Q-learning tradicional que a momentos iguala sus resultados y en tercer lugar policy gradient con una pendiente de aprendizaje menos pronunciada pero creciente en el tiempo, este último aprende más lentamente y obtiene buenos resultados pero en el rango de iteraciones observado no logra superar a las otras 2 metodologías, se deduce que debido al modo de entrenamiento (actualización de política una vez terminado un juego) la forma de aprender es más lenta pues requiere muchos más movimientos y tiempo, a diferencia de las metodologías de Q-Learning, que se actualizan para cada acción-estado.

* Comparativo de desempeño después de entrenamiento:
![comparación 10 juegos Humano vs Máquina luego de 1000 iteraciones de entrenamiento](https://github.com/FRo92/reinforcement_learning_project/blob/main/results_images/resultados_4.png)

Todas las implementaciones realizadas son capaces de ganarle a un humano en una secuencia de 10 juegos luego de un entrenamiento de 1000 juegos de aprendizaje.  Sin embargo, el ganador es Deep learning con un puntaje difícilmente lograble por un ser humano.

También se observó que los obstáculos del juego van variando en función del tiempo, debido a que mientrás mas tiempo pasa y más premios come la serpiente, su cuerpo se hace cada vez más largo, aumentando la complejidad de los movimientos, puesto que ahora no solo se deben evitar los bordes sino el cuerpo y cola del mismo agente (serpiente), en estos casos el desempeño de algoritmos de Q-Learning y Deep Q-Learning no era bueno, tienden a morir chocando con su mismo cuerpo a medida que pasa el tiempo, sin embargo,  el método de Policy Gradient si logra aprender una política en esas situaciones, logrando ejecutar movimientos claros que evitan tocar su propio cuerpo, a continuación se presenta la animación de la implementación de policy gradient donde se aprencia la adopción de la política para evitar lo descrito:

![Performance Snake con Policy Gradient](https://github.com/FRo92/reinforcement_learning_project/blob/main/results_images/gif_policy_gradient.gif)


## Conclusiones 👇🏼 ##

La técnica que logra el mejor desempeños es Deep Q-learning 🥇, sin embargo, técnicas más antiguas como q learning tradicional 🥈 logran un resultado muy cercano.
La técnica que aprende más rápido es el Q-learning tradicional y la más lenta (hasta 4 veces más lenta) es Policy gradient 🥉 posiblemente porque no actualiza la política hasta que la serpiente muere, sin embargo, es la que aprende de forma más estable, debido a que la forma de actualización de la política asegura de mejor forma ejecutar los movimientos que generen la mayor recompensa.

En uso de recursos computacionales policy gradient es la técnica más pesada con un uso de recursos 3 a 4 veces mayor que las otras 2 técnicas, posiblemente porque en la medida que va aprendiendo la función objetivo se vuelve más compleja.

Respecto a la facilidad de implementación, son similares ya que la mayor dificultad está en programar un entorno que refleje fielmente el problema que se quiere resolver.
Todos los algoritmos analizados logran un nivel de desempeño que supera a un ser humano, generando jugadas difícilmente igualables por un ser humano, sin embargo, el ser humano es capaz de aprender más rápido y con menos ejemplos.




