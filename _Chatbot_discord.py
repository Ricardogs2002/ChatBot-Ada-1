###Chat final con Discord###
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import json
import random
import tensorflow as tf
import pickle
import discord
import nest_asyncio
nest_asyncio.apply()

keyds='MTExODY5Njk5MjYyMTcyNzc5NA.G9MZgy.XQrENZHiW_sIzhka_Cb6STsaF56UJzrndsDPkg'
stemmer = LancasterStemmer()

with open("contenido.json", encoding='utf-8') as archivo:
    datos = json.load(archivo)

try:
    with open("variables.pickle", "rb") as archivoPickle:
        palabras, tags, entrenamiento, salida = archivoPickle.load(archivoPickle)
except:
    palabras = []
    tags = []
    auxX = []
    auxY = []

    for contenido in datos["contenido"]:
        for patrones in contenido["patrones"]:
            auxPalabra = nltk.word_tokenize(patrones)
            palabras.extend(auxPalabra)
            auxX.append(auxPalabra)
            auxY.append(contenido["tag"])

            if contenido["tag"] not in tags:
                tags.append(contenido["tag"])

    palabras = [stemmer.stem(w.lower()) for w in palabras if w != "?"]
    palabras = sorted(list(set(palabras)))
    tags = sorted(tags)

    entrenamiento = []
    salida = []
    salidaVacia = [0 for _ in range(len(tags))]

    for x, documento in enumerate(auxX):
        cubeta = []
        auxPalabra = [stemmer.stem(w.lower()) for w in documento]
        for w in palabras:
            if w in auxPalabra:
                cubeta.append(1)
            else:
                cubeta.append(0)
        filaSalida = salidaVacia[:]
        filaSalida[tags.index(auxY[x])] = 1
        entrenamiento.append(cubeta)
        salida.append(filaSalida)

    entrenamiento = np.array(entrenamiento)
    salida = np.array(salida)

    with open("variables.pickle", "wb") as archivoPickle:
        pickle.dump((palabras, tags, entrenamiento, salida), archivoPickle)

tf.compat.v1.reset_default_graph()

red = tflearn.input_data(shape=[None, len(entrenamiento[0])])
red = tflearn.fully_connected(red, 10)
red = tflearn.fully_connected(red, 10)
red = tflearn.fully_connected(red, len(salida[0]), activation="softmax")
red = tflearn.regression(red)

modelo = tflearn.DNN(red)

try:
    modelo.load("modelo.tflearn")
except tf.errors.NotFoundError:
    modelo.fit(entrenamiento, salida, n_epoch=1500, batch_size=150, show_metric=True)
    modelo.save("modelo.tflearn")


def mainBot():
  global keyds
  cliente=discord.Client()
  while True:
    @cliente.event
    async def on_message(mensaje):
      if mensaje.author == cliente.user:
        return

      #try:
        #entrada = input("Tu: ")
      cubeta = [0 for _ in range(len(palabras))]
      entradaProcesada = nltk.word_tokenize(mensaje.content)
      entradaProcesada = [stemmer.stem(palabra.lower()) for palabra in entradaProcesada]
      for palabra in entradaProcesada:
          for i, word in enumerate(palabras):
              if word == palabra:
                  cubeta[i] = 1
      resultados = modelo.predict([np.array(cubeta)])
      resultadosIndices = np.argmax(resultados)
      tag = tags[resultadosIndices]

      for tagAux in datos["contenido"]:
          if tagAux["tag"] == tag:
                respuesta = tagAux["respuestas"]

      #print("BOT:", random.choice(respuesta))
      await mensaje.channel.send(random.choice(respuesta))

      #except KeyboardInterrupt:
        # print("Saliendo...")
        #break
      #except Exception as e:
          #print("Se produjo un error:", str(e))
        #break
    cliente.run(keyds)

mainBot()
