import tensorflow as tf

# Define el modelo
input_dim = 784
output_dim = 10

inputs = tf.keras.Input(shape=(input_dim,))
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(output_dim, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Carga los pesos del archivo de checkpoint
checkpoint= '/sample_data/modelo.tflearn'
model.load_weights(checkpoint)

# Imprime los valores para verificar
print("Pesos cargados desde el archivo de checkpoint:", checkpoint)
