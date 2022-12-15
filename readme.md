# Técnicas de Reinforcement Learning aplicado al juego Snake 🐍 #

Documentación del código utilizado para desarrollar el proyecto final del curso Deep Learning Avanzado @ MIA UC, titulado "Benchmark de metodologías de Aprendizaje reforzado para Snake Game"

## Integrantes: ##
- José Antonio Lipari
- José Francisco Mallea
- Francisca Rojas
- Damaris Saavedra

A continuación se presenta el código usado para realizar el benchmark de las 3 técnicas de aprendizaje reforzado aplicado al juego Snake, cada carpeta alberga una metodología distinta, además se dejaron los archivos csv que contienen la información de las 2000 iteraciones que se compararon y presentaron en el informe final.

## Consideraciones: ##
Se trabajaron sobre códigos existentes en la web, presentados a continuación:
* [Policy Gradient](https://gist.github.com/ViniTheSwan/66fd59d78e94e06e00595ae9c1748d10#file-reinforce-py)
* [deep_q_learning](https://github.com/vedantgoswami/SnakeGameAI)
* [q_learning](https://gist.github.com/jl4r1991)

Se trabajó principalmente en la homologación de las condiciones del entorno para que sean comparables (tamaño y forma del tablero, tamaños de serpiente y manzana) dejando un tablero de 600x400 pixeles o bien 30x20 cuadrantes para cada implementación y además homologamos los movimientos de la serpiente.

### ¿Cómo ejecutar el código? 💻 ###
Primero se deben instalar las librerías necesarias (se recomienda crear un ambiente seguro) ejecutando el siguiente comando en la terminal (python -v 3.8.10):

```
pip install -r requirements.txt
```
Luego, existen instrucciones para cada implementación para iterar ~2000 veces el juego

## Deep Q Learning 🥇 ##
Para el método dql ejecutamos la instrucción:

```
make deep_q_learning
```
Mientras se ejecuta el juego se desplegará una ventana emergente con la interfaz del juego que permite ver el proceso de aprendizaje, los valores de cada juego quedarán impresos en la terminal.

## Q Learning 🥈 ##
Para inicializar este método primero debemos inicializar los valores de Q y luego ejecutar el agente:
```
make q_learning
```
Mientras se ejecuta el juego se desplegará una ventana emergente con la interfaz del juego que permite ver el proceso de aprendizaje, los valores de cada juego quedarán impresos en la terminal.

## Policy Gradient 🥉 ##
Para el método dql ejecutamos la instrucción:
```
make policy_gradient
```
Este método guarda los valores de cada juego en un archivo .csv y va guardando gifs cada vez que se alcanza un record de recomensas, quedan en la carpeta scores y gifs respectivamente.
