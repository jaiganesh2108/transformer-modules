import tensorflow as tf

def self_attention(input_shape, prefix='att', mask=False, **kwargs):
    inputs = tf.keras.layers.Input(shape=input_shape, dtype='float32', name=f"{prefix}_in1")
    attention = tf.keras.layers.MultiHeadAttention(name=f"{prefix}_att1", **kwargs)
    norm = tf.keras.layers.LayerNormalization(name=f'{prefix}_norm1')
    add = tf.keras.layers.Add(name=f'{prefix}_add1')
    
    attout = attention(query=inputs, value=inputs, key=inputs, use_causal_mask=mask)
    output = norm(add([inputs, attout]))

    model = tf.keras.Model(inputs=inputs, outputs=output, name=f"{prefix}_att")
    return model

def cross_attention(input_shape, context_shape, prefix='att', **kwargs):
    context = tf.keras.layers.Input(shape=context_shape, dtype='float32', name=f"{prefix}_ctx2")
    inputs = tf.keras.layers.Input(shape=input_shape, dtype='float32', name=f'{prefix}_in2')

    attention = tf.keras.layers.MultiHeadAttention(name=f'{prefix}_att2', **kwargs)
    norm = tf.keras.layers.LayerNormalization(name=f'{prefix}_norm2')
    add = tf.keras.layers.Add(name=f'{prefix}_add2')

    attout = attention(query=inputs, key=context, value=context)
    output = norm(add([attout, inputs]))

    model = tf.keras.Model(inputs=[context, inputs], outputs=output, name=f'{prefix}_crs_at')
    return model

def feed_forward(input_shape, model_dim, ff_dim, dropout=.1, prefix='ff'):
    inputs = tf.keras.layers.Input(shape=input_shape, dtype='float32', name=f'{prefix}_in3')

    dense1 = tf.keras.layers.Dense(ff_dim, name=f'{prefix}_ff1', activation='relu')
    dense2 = tf.keras.layers.Dense(model_dim, name=f'{prefix}_ff2')
    drop = tf.keras.layers.Dropout(dropout, name=f'{prefix}_drop')
    add = tf.keras.layers.Add(name=f"{prefix}_add3")

    ffout = drop(dense2(dense1(inputs)))

    norm = tf.keras.layers.LayerNormalization(name=f'{prefix}_norm3')
    output = norm(add([inputs, ffout]))

    model = tf.keras.Model(inputs=inputs, outputs=output, name=f'{prefix}_ff')

    return model
