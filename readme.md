# T茅cnicas de Reinforcement Learning aplicado al juego Snake 馃悕 #

Documentaci贸n del c贸digo utilizado para desarrollar el proyecto final del curso Deep Learning Avanzado @ MIA UC, titulado "Benchmark de metodolog铆as de Aprendizaje reforzado para Snake Game"

## Integrantes: ##
- Jos茅 Antonio Lipari
- Jos茅 Francisco Mallea
- Francisca Rojas
- Damaris Saavedra

A continuaci贸n se presenta el c贸digo usado para realizar el benchmark de las 3 t茅cnicas de aprendizaje reforzado aplicado al juego Snake, cada carpeta alberga una metodolog铆a distinta, adem谩s se dejaron los archivos csv que contienen la informaci贸n de las 2000 iteraciones que se compararon y presentaron en el informe final.

## Consideraciones: ##
Se trabaj贸 sobre c贸digo desarrollado por otros autores disponibles en la web, referenciados a continuaci贸n:
* [Policy Gradient](https://gist.github.com/ViniTheSwan/66fd59d78e94e06e00595ae9c1748d10#file-reinforce-py)
* [deep_q_learning](https://github.com/vedantgoswami/SnakeGameAI)
* [q_learning](https://gist.github.com/jl4r1991)

La ejecuci贸n y adopci贸n de los c贸digos anteriormente se帽alados no es trivial, puesto que carecen de documentaci贸n y algunas funciones se encuentran deprecadas, por lo que gran parte del tiempo se invirti贸 en entender los c贸digos y lograr hacerlos funcionar en nuestras m谩quinas locales.

Se implement贸 un ambiente virtual seguro donde la ejecuci贸n de los tres m茅todos fuera compatible, los requerimientos de este ambiente se encuentran disponibles en el archivo [requeriments.txt](https://github.com/FRo92/reinforcement_learning_project/blob/main/requeriments.txt) de este repositorio.

Tambi茅n, se trabaj贸 en la homologaci贸n de las condiciones del entorno para que sean comparables (tama帽o y forma del tablero, tama帽os de serpiente y manzana) dejando un tablero de 600x400 pixeles o bien 30x20 cuadrantes para cada implementaci贸n, velocidad de simulaci贸n y adem谩s homologamos los movimientos de la serpiente.

Finalmente, decidimos implementar un m茅todo m谩s para realizar una comparaci贸n que resulte m谩s evidente, esta es la comparaci贸n con un jugador [humano](https://www.linkedin.com/search/results/all/?heroEntityKey=urn%3Ali%3Afsd_profile%3AACoAACtV7soBYqUskGfD1tRhBlnIzp756eJ64xg&keywords=jos%C3%A9%20antonio%20lipari%20arias&origin=RICH_QUERY_TYPEAHEAD_HISTORY&position=0&searchId=78036010-c4b8-4767-97f9-a1243137a39a&sid=%409V) sin experiencia previa en el juego.

### 驴C贸mo ejecutar el c贸digo? 馃捇 ###
Primero se deben instalar las librer铆as necesarias (se recomienda crear un ambiente seguro) ejecutando el siguiente comando en la terminal (python -v 3.8.10):

```
pip install -r requirements.txt
```
Luego, existen instrucciones para cada implementaci贸n para iterar ~2000 veces el juego

## Deep Q Learning 馃 ##

Es el [m茅todo](https://github.com/FRo92/reinforcement_learning_project/tree/main/deep_q_learning) m谩s actual, se basa en el q_learning tradicional, es decir, se busca resolver la ecuaci贸n de Bellman, pero se busca representar la funci贸n Q con una red neuronal. En este caso se implement贸 en pytorch una red de una capa oculta de tama帽o 256.

Esta implementaci贸n requiere tunear los par谩metros: learning rate (factor de aprendizaje), penalizaci贸n (factor de olvido) y epsilon (factor de exploraci贸n incial), adem谩s, primero se ejecuta el agente, el que a su vez llama al modelo y a la interfaz.

```
make dql
```

Mientras se ejecuta el juego se desplegar谩 una ventana emergente con la interfaz del juego que permite ver el proceso de aprendizaje, los valores de cada juego quedar谩n impresos en la terminal.

## Q Learning 馃 ##

Para el m茅todo [dql](https://github.com/FRo92/reinforcement_learning_project/tree/main/q_learning_tradicional) es el m谩s antiguo de la comparaci贸n, se basa en la soluci贸n de la ecuaci贸n de Bellman:

$$
Q(s,a) = R(s,a) + \lambda max_{a'\in A}Q(s',a')
$$

Donde Q(s,a) es el valor que se busca llenar de forma tabular en funci贸n de estados s y acciones a, $\lambda$ es el factor de olvido de recompensas lejanas y $max_{a'\in A}Q(s',a')$ es la acci贸n que maximiza la recompensa futura.

Esta implementaci贸n requiere ajustar los par谩metros $\lambda$ y el factor de exploraci贸n $\epsilon$.

Para ejecutar este m茅todo, debemos primero debemos inicializar los valores de Q y luego ejecutar el agente con la siguiente instrucci贸n:
```
make q_learning
```

Mientras se ejecuta el juego se desplegar谩 una ventana emergente con la interfaz del juego que permite ver el proceso de aprendizaje, los valores de cada juego quedar谩n impresos en la terminal.

## Policy Gradient 馃 ##

El m茅todo [policyGradient](https://github.com/FRo92/reinforcement_learning_project/tree/main/policy_gradient), a diferencia de los m茅todos anteriores, busca aprender la pol铆tica directamente como una funci贸n de probabilidad, en este caso, discreta usando [tfp.distributions.Categorical](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Categorical). Adem谩s, la pol铆tica se actualiza cada vez que la serpiente muere. Se busca maximizar la probabilidad de aquellas acciones que aumentan la recompenza y minimizar aquellas que la disminuyen, lo que se traduce en la siguiente funci贸n de p茅rdida:

$$
\ell(\theta) = -log \pi(a_t | s_t)G_t 
$$

Donde $\pi$ es la pol铆tica y $G_t$ es la recompensa total.
Aplicando el descenso de gradiente a la funci贸n de p茅rdida se obtiene:

$$
\theta \leftarrow \theta + \eta \nabla log \pi(a_t | s_t)G_t 
$$

Esta implementaci贸n est谩 hecha en tenssorflow con dos capas densas de tama帽o 30.

Para el m茅todo pg primero s贸lo se ejecuta el agente snake.py, que llama al modelo PolicyGradient mediante la instrucci贸n:
```
make pg
```
Este m茅todo guarda los valores de cada juego en un archivo .csv y va guardando gifs cada vez que se alcanza un record de recomensas, quedan en la carpeta scores y gifs respectivamente.

## Desaf铆os de preparaci贸n y homologaci贸n 鈿欙笍 ##
Se investigan 8 implementaciones realizadas por autores diferentes, de las cuales se seleccionaron 3 las que fueron implementadas con las siguientes consideraciones:
* Habilitaci贸n hardware propio ya que no ejecutan en colab las implementaciones utilizadas, por problemas de versionamiento o interfaz gr谩fica.
* GPU utilizada Nvidia: GTX 1660TI  16GB ram en entorno
* CPU utilizada: intel I9                     16GB ram en entono
* Instalaci贸n y aprendizaje de librer铆as:
Pygame /Pytorch /imageio/scipy.ndimage/Tensorflow/matplotlib.pyplot
Implementaci贸n de entornos virtuales espec铆ficos para cada t茅cnica  ya que las versiones GPU presentaban conflictos de instalaci贸n:
* Se utiliz贸 Anaconda instalando versiones espec铆ficas de cada implementaci贸n.

## M茅tricas de comparaci贸n 鈿栵笍 ##
Para poder realizar una comparaci贸n efectiva debemos definir un entorno de comparaci贸n justo y modificar las diversas implementaciones para que sean comparables entre ellas, con ese objetivo tomamos las siguientes consideraciones:
Ajustamos el tama帽os del tablero a 30x20 cuadrantes o movimientos en todas las implementaciones
Comparamos mejoras en puntaje obtenido vs cantidad de juegos de entrenamiento para definir qu茅 algoritmo aprende m谩s r谩pido comparando puntaje promedio y m谩ximo.

Incorporamos datos de un humano aprendiendo en condiciones similares como referencia
Adicionalmente, comparamos el tiempo de implementaci贸n, ya que realizar un juego demora tiempos distintos dependiendo del algoritmo utilizado.

Finalmente, preparamos versiones distribu铆das utilizando CUDA y versiones no distribuidas usando CPU para comparar los beneficios en t茅rminos de ahorro de tiempo de entrenamiento, sin embargo, el c贸digo presentado en este repositorio solo presenta la implementaci贸n en CPU.

## Resultados 馃搱 ##
A continuaci贸n, se presentan los resultados obtenidos al comparar el rendimiento de las tres metodolog铆as:
* Resultados comparaci贸n Humano vs Aprendizaje reforzado para 60 iteraciones:
![comparaci贸n 60 iteraciones Humano vs M谩quina](https://github.com/FRo92/reinforcement_learning_project/blob/main/results_images/resultados_1.png)

Para tener un punto de comparaci贸n y referencia entrenamos un [humano](https://www.linkedin.com/search/results/all/?heroEntityKey=urn%3Ali%3Afsd_profile%3AACoAACtV7soBYqUskGfD1tRhBlnIzp756eJ64xg&keywords=jos%C3%A9%20antonio%20lipari%20arias&origin=RICH_QUERY_TYPEAHEAD_HISTORY&position=0&searchId=78036010-c4b8-4767-97f9-a1243137a39a&sid=%409V) sin experiencia previa en el juego y registramos c贸mo evoluciona su puntaje respecto a los algoritmos de aprendizaje reforzado (2 bloques de 45 minutos de entrenamiento).

Se observa que el humano aprende mucho m谩s r谩pido que cualquiera de los m茅todos probados cuando se comparan solo 60 iteraciones, registrando un m谩ximo de 41 puntos  y un promedio de 31 puntos una vez que aprende. De ac谩 surge la pregunta, 驴cu谩ntas iteraciones m谩s que un humano requiere una sistema para llegar o incluso sobrepasar el desempe帽o humano?

* Resultados comparaci贸n Humano vs Aprendizaje reforzado para 1800 iteraciones:
![comparaci贸n 1800 iteraciones Humano vs M谩quina](https://github.com/FRo92/reinforcement_learning_project/blob/main/results_images/resultados_3.png)

Se puede ver que el mejor desempe帽o lo obtiene algoritmo deep Q learning despu茅s de la iteraci贸n 300, seguido de cerca por Q-learning tradicional que a momentos iguala sus resultados y en tercer lugar policy gradient con una pendiente de aprendizaje menos pronunciada pero creciente en el tiempo, este 煤ltimo aprende m谩s lentamente y obtiene buenos resultados pero en el rango de iteraciones observado no logra superar a las otras 2 metodolog铆as, se deduce que debido al modo de entrenamiento (actualizaci贸n de pol铆tica una vez terminado un juego) la forma de aprender es m谩s lenta pues requiere muchos m谩s movimientos y tiempo, a diferencia de las metodolog铆as de Q-Learning, que se actualizan para cada acci贸n-estado.

* Comparativo de desempe帽o despu茅s de entrenamiento:
![comparaci贸n 10 juegos Humano vs M谩quina luego de 1000 iteraciones de entrenamiento](https://github.com/FRo92/reinforcement_learning_project/blob/main/results_images/resultados_4.png)

Todas las implementaciones realizadas son capaces de ganarle a un humano en una secuencia de 10 juegos luego de un entrenamiento de 1000 juegos de aprendizaje.  Sin embargo, el ganador es Deep learning con un puntaje dif铆cilmente lograble por un ser humano.

Tambi茅n se observ贸 que los obst谩culos del juego van variando en funci贸n del tiempo, debido a que mientr谩s mas tiempo pasa y m谩s premios come la serpiente, su cuerpo se hace cada vez m谩s largo, aumentando la complejidad de los movimientos, puesto que ahora no solo se deben evitar los bordes sino el cuerpo y cola del mismo agente (serpiente), en estos casos el desempe帽o de algoritmos de Q-Learning y Deep Q-Learning no era bueno, tienden a morir chocando con su mismo cuerpo a medida que pasa el tiempo, sin embargo,  el m茅todo de Policy Gradient si logra aprender una pol铆tica en esas situaciones, logrando ejecutar movimientos claros que evitan tocar su propio cuerpo, a continuaci贸n se presenta la animaci贸n de la implementaci贸n de policy gradient donde se aprencia la adopci贸n de la pol铆tica para evitar lo descrito:

![Performance Snake con Policy Gradient](https://github.com/FRo92/reinforcement_learning_project/blob/main/results_images/episode_683.gif)


## Conclusiones 馃憞馃徏 ##

La t茅cnica que logra el mejor desempe帽os es Deep Q-learning 馃, sin embargo, t茅cnicas m谩s antiguas como q learning tradicional 馃 logran un resultado muy cercano.
La t茅cnica que aprende m谩s r谩pido es el Q-learning tradicional y la m谩s lenta (hasta 4 veces m谩s lenta) es Policy gradient 馃 posiblemente porque no actualiza la pol铆tica hasta que la serpiente muere, sin embargo, es la que aprende de forma m谩s estable, debido a que la forma de actualizaci贸n de la pol铆tica asegura de mejor forma ejecutar los movimientos que generen la mayor recompensa.

En uso de recursos computacionales policy gradient es la t茅cnica m谩s pesada con un uso de recursos 3 a 4 veces mayor que las otras 2 t茅cnicas, posiblemente porque en la medida que va aprendiendo la funci贸n objetivo se vuelve m谩s compleja.

Respecto a la facilidad de implementaci贸n, son similares ya que la mayor dificultad est谩 en programar un entorno que refleje fielmente el problema que se quiere resolver.
Todos los algoritmos analizados logran un nivel de desempe帽o que supera a un ser humano, generando jugadas dif铆cilmente igualables por un ser humano, sin embargo, el ser humano es capaz de aprender m谩s r谩pido y con menos ejemplos.




