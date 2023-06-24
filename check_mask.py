import io
import os

import cv2
import scipy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm


def calculate_iou(gt_mask, pred_mask):
    gt_mask[gt_mask > 0] = 1
    pred_mask[pred_mask > 0] = 1
    overlap = pred_mask * gt_mask
    union = (pred_mask + gt_mask) > 0
    iou = overlap.sum() / float(union.sum())
    return iou


def load_model(model_dir):
    architecture_path = os.path.join(model_dir, 'architecture.json')
    weights_path = os.path.join(model_dir, 'best_weights.h5')
    with open(architecture_path, 'r') as f:
        model = tf.keras.models.model_from_json(f.read())
    model.load_weights(weights_path)

    model.compile(
        optimizer='adam',
        loss='bce',
    )
    return model


if __name__ == '__main__':
    model_name = 'Xception'

    model = load_model('models/covid-19 classification#tensorflow#Xception#test_time=04-07-20.11')
    # from tensorflow.keras.models import load_model

    # model = tf.saved_model.load('models/covid-19 classification#tensorflow#Xception#test_time=02-22-22.42/saved_model')
    # tf.keras.models.save_model(model, 'models/covid-19 classification#tensorflow#Xception#test_time=02-22-22.42/saved_model_new/')

    # model = tf.keras.models.load_model('models/covid-19 classification#tensorflow#Xception#test_time=02-22-22.42/saved_model')
    # model = load_model('models/covid-19 classification#tensorflow#Xception#test_time=02-22-22.42/saved_model')
    last_weight = model.layers[-1].get_weights()[0]  # (1280, 2)
    new_model = tf.keras.Model(
        inputs=model.input,
        outputs=(
            model.layers[-5].output,  # the layer just before GAP, for using spatial features
            model.layers[-1].output
        )
    )

    DATA_DIR = '/home/vladislav/PythonProject/Datasets/QaTa-COV19/QaTa-COV19-v2/Val Set'
    for img_path in tqdm(glob(f'{DATA_DIR}/img/*.png')):

        save_dir = f'results/{model_name}/{os.path.splitext(os.path.basename(img_path))[0]}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        img_string = tf.io.read_file(img_path)
        img_input = tf.image.decode_image(img_string, channels=3)
        img_resized = tf.image.resize(images=img_input, size=(448, 448))
        img_norm = img_resized / 255.0
        img_output = tf.reshape(tensor=img_norm, shape=(448, 448, 3))

        last_conv_output, pred = new_model(np.array([img_output]))
        last_conv_output = np.squeeze(last_conv_output)  # (14, 14, 2048)
        feature_activation_maps = scipy.ndimage.zoom(last_conv_output, (32, 32, 1),
                                                     order=1)  # (14, 14, 2048) => (448, 448, 2048)
        pred_class = np.argmax(pred)
        print(pred)  # 0: Full, 1: Free
        predicted_class_weights = last_weight[:, pred_class]  # (1280, 1)
        final_output = np.dot(feature_activation_maps.reshape((448 * 448, 2048)), predicted_class_weights).reshape(
            (448, 448))  # (224*224, 1280) dot_produt (1280, 1) = (224*224, 1)

        plt.pcolormesh(final_output, cmap='plasma')
        plt.axis('equal')
        plt.axis('off')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        mask = cv2.imdecode(img_arr, 1)
        mask = mask[15:-15, 95: -95]
        mask = cv2.resize(mask, (448, 448))

        # 1. Feature_matrix
        cv2.imwrite(f'{save_dir}/feature_matrix.png', mask)

        # 2. Input_img
        img_input = np.array(cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)).astype(np.uint8)
        cv2.imwrite(f'{save_dir}/input_img.png', img_input)

        # 3. Input_img union feature_matrix
        mask_union = cv2.addWeighted(np.array(img_input).astype(np.uint8), 1, np.array(mask).astype(np.uint8), 0.6, 0)
        cv2.imwrite(f'{save_dir}/feature_matrix_union.png', mask_union)

        # Color_mask
        # img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # ret, mask = cv2.threshold(img2gray, 120, 255, cv2.THRESH_BINARY)
        #
        # color_mask = np.zeros(img2.shape)
        # color_mask[mask == 0] = (128, 128, 128)
        # color_mask[mask == 255] = (33, 45, 105)
        # cv2.imwrite('result/color_mask.png', color_mask)

        # 4. Input_img union plasma mask
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask_full = cv2.threshold(mask_gray, 120, 255, cv2.THRESH_BINARY)
        img2_fg = cv2.bitwise_and(mask, mask, mask=mask_full)
        dst = cv2.addWeighted(img_input, 1, np.array(img2_fg).astype(np.uint8), 0.45, 0)
        cv2.imwrite(f'{save_dir}/plasma_mask_union.png', dst)

        # 5. Input_img union mask
        dst_1 = cv2.addWeighted(np.array(img_input), 1, (cv2.cvtColor(np.array(mask_full).astype('uint8'), cv2.COLOR_GRAY2BGR)
                                                    * (50, 125, 210)).astype('uint8'), 0.45, 0)
        cv2.imwrite(f'{save_dir}/color_mask_union.png', dst_1)

        # 6. Gray mask
        cv2.imwrite(f'{save_dir}/mask_pred.png', mask_full)
        ann = cv2.imread(f'{img_path.replace("img/", "ann/mask_")}', 0)
        ann = cv2.resize(ann, (448, 448))
        cv2.imwrite(f'{save_dir}/mask_real.png', ann)

        iou = calculate_iou(
            ann,
            mask_full,
        )
        print(iou)
