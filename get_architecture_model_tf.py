import tensorflow as tf

from typing import List, Tuple


def get_model(
    input_shape: Tuple[int, int, int],
    output_size: int,
    activation_func: str,
    encoder_name: str,
    dropout_size: float,
    regularize_size: float,
    use_trainable_model: bool,
    head_cnn: List[int],
    label: str,
):

    encoder = eval(f'tf.keras.applications.{encoder_name}')(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
    )
    encoder.trainable = use_trainable_model
    for layer in encoder.layers[-10:]:
        layer.trainable = True
    x = encoder.output
    x = tf.keras.layers.Dropout(rate=dropout_size)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    for size in head_cnn:
        x = tf.keras.layers.Dense(size, activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout_size)(x)

    predictions = tf.keras.layers.Dense(
        output_size,
        name=label,
        activation=activation_func,
    )(x)

    model = tf.keras.Model(
        inputs=encoder.input,
        outputs=predictions,
    )
    model.summary()

    return model


def get_model_new(
    input_shape: Tuple[int, int, int],
    output_size: List[int],
    activation_func: List[str],
    labels: List[str],
    encoder_name: str,
    dropout_size: float,
    regularize_size: float,
    use_trainable_model: bool,
):

    encoder = eval(f'tf.keras.applications.{encoder_name}')(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
    )
    encoder.trainable = False
    x = encoder.output
    x = tf.keras.layers.Dropout(rate=dropout_size)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # for size in head_cnn:
    #     x = tf.keras.layers.Dense(size, activation='relu')(x)
    #     x = tf.keras.layers.Dropout(dropout_size)(x)

    predictions = []
    for units, act, label in zip(output_size, activation_func, labels):
        y = tf.keras.layers.Dense(1024, activation='relu')(x)
        y = tf.keras.layers.Dropout(dropout_size)(y)
        predictions.append(tf.keras.layers.Dense(
            units,
            activation=act,
            name=label,
        )(y))

    model = tf.keras.Model(
        inputs=encoder.input,
        outputs=predictions,
    )
    model.summary()

    return model


if __name__ == '__main__':
    model = get_model(
        input_shape=(448, 448, 3),
        output_size=2,
        activation_func='sigmoid',
        # encoder_name='ResNet50',
        # encoder_name='EfficientNetB5',
        encoder_name='convnext.ConvNeXtBase',
        dropout_size=0.3,
        regularize_size=0.001,
        use_trainable_model=False,
        head_cnn=[1024],
        label='output'
    )

    # model = get_model_new(
    #     input_shape=(224, 224, 3),
    #     output_size=[4, 100, 150, 99],
    #     activation_func=['softmax', 'softmax', 'softmax', 'softmax'],
    #     labels=['category', 'fruits', 'dish', 'drinks'],
    #     encoder_name='Xception',
    #     dropout_size=0.3,
    #     regularize_size=0.001,
    #     use_trainable_model=False,
    #     # head_cnn=[1024, 512]
    # )
    with open('architecture.json', 'w') as f:
        f.write(model.to_json())