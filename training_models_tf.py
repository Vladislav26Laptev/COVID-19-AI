import os
import time
import json
import wandb
import shutil
import argparse
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

from glob import glob
from typing import List
from wandb.keras import WandbCallback
from sklearn.model_selection import train_test_split

from utils.get_architecture_model_tf import get_model

os.environ['WANDB_API_KEY'] = '0a94ef68f2a7a8b709671d6ef76e61580d20da7f'

METRICS = {
    'logcosh': tf.keras.losses.LogCosh(),
    'mae': tf.keras.losses.MeanAbsoluteError(),
    'mse': tf.keras.losses.MeanSquaredError(),
    'bce': tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0),
    # 'cce': tf.keras.metrics.CategoricalCrossentropy(name='categorical_crossentropy', dtype=None, from_logits=False,
    #                                                 label_smoothing=0, axis=-1),
    'ch': tf.keras.metrics.CategoricalHinge(name='categorical_hinge', dtype=None),
    'precision': tf.keras.metrics.Precision(top_k=None, thresholds=0.5, name='precision'),
    # 'precision_at_recall': tf.keras.metrics.PrecisionAtRecall(recall, num_thresholds=200, class_id=None, name=None,
    #                                                           dtype=None),
    'recall': tf.keras.metrics.Recall(top_k=None, thresholds=0.5, name='recall'),
    'b_auc': tf.keras.metrics.AUC(name='auc_roc', num_thresholds=200, curve='ROC'),
    'c_auc': tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy', dtype=None),

    'f1': tfa.metrics.F1Score(name='f1_score', num_classes=2),
}


def get_inputs_and_targets(
        data_path: str,
        classes: List[str],
):
    inputs_path, targets_data = [], []
    if len(classes) == 0:
        classes = os.listdir(data_path)
    for idx, class_name in enumerate(classes):
        images_path = glob(f'{data_path}/{class_name}/*.[pj][npe][ge]*')
        targets = np.zeros(shape=(len(images_path), len(classes)))
        targets[:, idx] = 1
        inputs_path.extend(images_path)
        targets_data.extend(targets)

    return np.array(inputs_path), np.array(targets_data)


class DataSequence(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, img_height, img_width, num_channels, batch_size):
        self.x, self.y = x_set, y_set
        self.height, self.width, self.channels = img_height, img_width, num_channels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        paths = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        labels = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        images = self._processing(paths=paths,)
        batch_sample = images, np.array(labels)
        return batch_sample

    def _aug(self, img):
        return img

    def _processing(self, paths):
        hu_batch = np.zeros(shape=(len(paths), self.height, self.width, self.channels), dtype=np.float32)

        for idx, path in enumerate(paths):
            img_string = tf.io.read_file(path)
            img_input = tf.image.decode_image(img_string, channels=self.channels)
            img_input = self._aug(img_input)

            img_resized = tf.image.resize(images=img_input, size=(self.height, self.width))
            img_norm = img_resized / 255.0
            img_output = tf.reshape(tensor=img_norm, shape=(self.height, self.width, self.channels))

            hu_batch[idx] = img_output

        return hu_batch


class Net:
    def __init__(self, image_size, loss_function, optimizer_name, encoder_name, model_name, learning_rate, head_cnn,
                 classes, dropout_size, regularize_size, metrics, epochs, batch_size, train_path, use_trainable_model,
                 activation_func, class_weight, val_path):
        self.metrics = metrics
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.image_size = image_size
        self.encoder_name = encoder_name
        self.dropout_size = dropout_size
        self.regularize_size = regularize_size
        self.use_trainable_model = use_trainable_model
        self.classes = classes
        self.class_weight = class_weight
        self.model_dir = model_dir
        self.head_cnn = head_cnn
        self.model_name = model_name
        self.train_path = train_path
        self.val_path = val_path
        self.activation_func = activation_func

    # ----------------------------------------------------------------------------------------------------------------------
    def save_model(self, model):
        print('-' * 70)
        print('Saving the model and its weights...')
        start = time.time()
        with open(os.path.join(self.model_dir, 'architecture.json'), 'w') as f:
            f.write(model.to_json())
        end = time.time()
        print('Saving the model and its weights takes ({:1.2f} seconds)'.format(end - start))

    # ----------------------------------------------------------------------------------------------------------------------
    def load_model(self, model_dir):
        print('Loading the model and its weights...')
        start = time.time()
        architecture_path = os.path.join(model_dir, 'architecture.json')
        weights_path = os.path.join(model_dir, 'best_weights.h5')
        with open(architecture_path, 'r') as f:
            model = tf.keras.models.model_from_json(f.read())
        model.load_weights(weights_path)

        model.compile(
            optimizer=self.get_optimizer(),
            loss=self.loss_function,
            metrics=self.metrics
        )
        end = time.time()
        print('Loading the model and its weights takes ({:1.2f} seconds)'.format(end - start))
        return model

    # ----------------------------------------------------------------------------------------------------------------------
    def get_optimizer(self):
        if self.optimizer_name == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'adamax':
            opt = tf.keras.optimizers.Adamax(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'radam':
            opt = tfa.optimizers.RectifiedAdam(learning_rate=self.learning_rate)
        else:
            raise ValueError('Undefined OPTIMIZER_TYPE!')
        return opt

    # ----------------------------------------------------------------------------------------------------------------------
    def get_loss(self):
        if self.loss_function == 'logcosh':
            loss = tf.keras.losses.LogCosh()
        elif self.loss_function == 'binary_crossentropy':
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0)
        elif self.loss_function == 'mae':
            loss = tf.keras.losses.MeanAbsoluteError()
        elif self.loss_function == 'categorical_crossentropy':
            loss = tf.keras.metrics.CategoricalCrossentropy(name='categorical_crossentropy', dtype=None,
                                                            from_logits=False, label_smoothing=0, axis=-1),
        else:
            raise ValueError("Wrong loss")
        return loss

    # ----------------------------------------------------------------------------------------------------------------------
    def get_metrics(self):
        metrics = []
        for metric in self.metrics:
            metrics.append(METRICS[metric])
        return metrics

    # ----------------------------------------------------------------------------------------------------------------------
    def get_compile_model(self):
        model = get_model(
            encoder_name=self.encoder_name,
            input_shape=(int(self.image_size[0]), int(self.image_size[1]), int(self.image_size[2])),
            dropout_size=self.dropout_size,
            regularize_size=self.regularize_size,
            head_cnn=self.head_cnn,
            use_trainable_model=eval(self.use_trainable_model),
            output_size=len(self.classes),
            activation_func=self.activation_func,
            label='output_layer',
        )

        model.compile(
            optimizer=self.get_optimizer(),
            loss=self.get_loss(),
            # loss='categorical_crossentropy',
            metrics=self.get_metrics()
        )
        return model

    # ----------------------------------------------------------------------------------------------------------------------
    def train_model(self):

        # DATA loading
        x_train, y_train = get_inputs_and_targets(
            data_path=self.train_path,
            classes=self.classes,
        )

        x_val, y_val = get_inputs_and_targets(
            data_path=self.val_path,
            classes=self.classes,
        )

        ds_train = DataSequence(x_set=x_train, y_set=y_train, img_height=self.image_size[0], img_width=self.image_size[1],
                                batch_size=self.batch_size, num_channels=self.image_size[2])
        ds_val = DataSequence(x_set=x_val, y_set=y_val, img_height=self.image_size[0], img_width=self.image_size[1],
                              batch_size=self.batch_size, num_channels=self.image_size[2])

        with open(self.class_weight) as f:
            class_weight = json.load(f)
            f.close()

        # MODEL compile
        model = self.get_compile_model()
        # self.save_model(model)

        # Training compile pipeline
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=15)

        wandb_config = dict(
            image_size=self.image_size,
            loss_function=self.loss_function,
            optimizer_name=self.optimizer_name,
            encoder_name=self.encoder_name,
            model_name=self.model_name,
            learning_rate=self.learning_rate,
            head_cnn=self.head_cnn,
            classes=self.classes,
            dropout_size=self.dropout_size,
            regularize_size=self.regularize_size,
            epochs=self.epochs,
            batch_size=self.batch_size,
            use_trainable_model=self.use_trainable_model,
            activation_func=self.activation_func,
            framework='tensorflow',
        )

        wandb.init(project=self.model_name.split('#')[0],
                   dir=self.model_dir,
                   name=self.model_name,
                   config=wandb_config,
                   tags=[
                       wandb_config['encoder_name'],
                       str(wandb_config['use_trainable_model']),
                       wandb_config['framework'],
                   ],
                   sync_tensorboard=True)

        wb = WandbCallback(
            monitor='val_loss',
            mode='min',
            save_weights_only=True,
            save_model=True,
            log_evaluation=True,
        )

        filepath = os.path.join(self.model_dir + '/best_weights.h5')

        weight_saver = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                          monitor='val_loss',
                                                          verbose=1,
                                                          save_best_only=True,
                                                          save_weights_only=True,
                                                          mode='min')

        callbacks = [wb, weight_saver]

        print("\033[32m encoder: {}".format(self.encoder_name))
        print("\033[32m optimizer: {}".format(self.optimizer_name))
        print("\033[32m input_size: {}x{}".format(self.image_size[0], self.image_size[1]))
        print("\033[32m classes: {}".format(self.classes))
        print("\033[32m activation: {}".format(self.activation_func))
        print("\033[32m learning_rate: {}".format(self.learning_rate))
        print("\033[32m train_batch_size: {}".format(self.batch_size))
        print("\033[32m epochs: {}".format(self.epochs))

        class_weight = {
            0: 1.25,
            1: 1.0,
        }
        n_cpu = os.cpu_count()

        self.save_model(model)

        model.fit(
            ds_train,
            class_weight=class_weight,
            epochs=self.epochs,
            # verbose=1,
            validation_data=ds_val,
            callbacks=callbacks,
            workers=n_cpu,
        )
        tf.saved_model.save(
            model,
            f'{self.model_dir}/saved_model/',
        )


if __name__ == '__main__':
    with open('parameters.json') as f:
        parameters = json.load(f)
        f.close()

    parser = argparse.ArgumentParser(description='Run training model classification')
    parser.add_argument('-i', '--image_size', type=int, nargs='+', default=parameters['image_size'])
    parser.add_argument('-l', '--loss_function', type=str, default=parameters['loss_function'])
    parser.add_argument('-o', '--optimizer_name', type=str, default=parameters['optimizer_name'])
    parser.add_argument('-e', '--encoder_name', type=str, default=parameters['encoder_name'])
    parser.add_argument('-m', '--model_name', type=str, default=parameters['model_name'])
    parser.add_argument('-lr', '--learning_rate', type=float, default=parameters['learning_rate'])
    parser.add_argument('--head_cnn', type=int, nargs='+', default=parameters['head_cnn'])
    parser.add_argument('-d', '--dropout_size', type=float, default=parameters['dropout_size'])
    parser.add_argument('-c', '--classes', type=str, nargs='+', default=parameters['classes'])
    parser.add_argument('--class_weight', type=str, default=parameters['class_weight'])
    parser.add_argument('-a', '--activation_func', type=str, default=parameters['activation_func'])
    parser.add_argument('--regularize_size', type=float, default=parameters['regularize_size'])
    parser.add_argument('--metrics', type=str, nargs='+', default=parameters['metrics'])
    parser.add_argument('--epochs', type=int, default=parameters['epochs'])
    parser.add_argument('--batch_size', type=int, default=parameters['batch_size'])
    parser.add_argument('--train_path', type=str, default=parameters['train_path'])
    parser.add_argument('--val_path', type=str, default=parameters['val_path'])
    parser.add_argument('-u', '--use_trainable_model', type=str, default=parameters['use_trainable_model'])
    args = parser.parse_args()

    today = datetime.datetime.today()
    model_name = f'{args.model_name}#tensorflow#{args.encoder_name}#test_time={today.strftime("%m-%d-%H.%M")}'
    model_dir = os.path.join('models', model_name)

    net = Net(
        image_size=args.image_size,
        loss_function=args.loss_function,
        optimizer_name=args.optimizer_name,
        encoder_name=args.encoder_name,
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        head_cnn=args.head_cnn,
        classes=args.classes,
        class_weight=args.class_weight,
        dropout_size=args.dropout_size,
        regularize_size=args.regularize_size,
        metrics=args.metrics,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_path=args.train_path,
        val_path=args.val_path,
        use_trainable_model=args.use_trainable_model,
        activation_func=args.activation_func,
    )

    shutil.rmtree(model_dir, ignore_errors=True)
    os.makedirs(model_dir)
    net.train_model()