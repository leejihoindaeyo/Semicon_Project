#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import time
import json
import numpy as np
import argparse
import cv2
from hoonpyutils import UtilsCommon as utils
from hoonpyutils import UtilsImage  as uImg

import multiprocessing

import traceback

__author__ = "Kim, Jeonguk"
__credits__ = ["Kim, Jeonguk", "Paek, Hoon"]
__license__ = "GPL"
__version__ = "0.1.0"
__maintainer__ = ["Kim, Jeonguk", "Paek, Hoon"]
__status__ = "Release"      # Development / Test / Release.

_this_folder_ = os.path.dirname(os.path.abspath(__file__))
_this_basename_ = os.path.splitext(os.path.basename(__file__))[0]

IMG_PERS_SUB_FOLDER = 'check_img'
JSON_PERS_SUB_FOLDER = 'Results'
IMAGE_FILE_EXT = '.jpg'
JSON_FILE_EXT = '.json'


class PersDetect(multiprocessing.Process):

    def __init__(self, desc="", ini=None, logger=None):
        super().__init__()

        self.desc = desc
        self.ini = ini
        self.method = None

        self.detect_inst = None

        self.logger = logger

        self.img_pers_dir = None
        self.json_pers_dir = None

        self.gpu_list = '1'

        if self.ini is not None:
            self.init_ini(ini['PERS_DETECT'])

    def init_ini(self, ini):
        self.gpu_list = ini['gpu_list'].strip()
        self.method = ini['method'].strip()

    def init_out_dirs(self, out_path):
        self.img_pers_dir = os.path.join(out_path, IMG_PERS_SUB_FOLDER) + "/"
        self.json_pers_dir = os.path.join(out_path, JSON_PERS_SUB_FOLDER) + "/"

        utils.check_directory_existence(self.img_pers_dir, create_=True, print_=False)
        utils.check_directory_existence(self.json_pers_dir, create_=True, print_=False)

    def init_detection_instance(self):
        self.logger.info(" # inference method initialization by {}".format(self.method))
        try:
            if self.method == 'CENTERNET':
                from CenterNet import centernet_class
                self.detect_inst = centernet_class.CenterNet(gpu_list=self.gpu_list,
                                                             ini=self.ini['CENTERNET'],
                                                             logger=self.logger)
            else:
                self.logger.error(" @ Incorrect inference method, {}".format(self.method))


        except Exception as e:
            self.logger.error(" # init_detection_instance.exception : {}".format(e))
            self.logger.error(traceback.format_exc())

    @staticmethod
    def calculate_bboxes_absolute_position(img_size, bboxes):
        h = img_size[0]
        w = img_size[1]
        abs_bboxes = []
        for bbox in bboxes:
            abs_bbox = [int(round(bbox[0]*w)), int(round(bbox[1]*h)), int(round(bbox[2]*w)), int(round(bbox[3]*h))]
            abs_bboxes.append(abs_bbox)

        return abs_bboxes

    @staticmethod
    def refine_aabb_to_clockwise_bboxes(bboxes):
        refine_bboxes = []
        for bbox in bboxes:
            refine_bbox = [[bbox[0], bbox[1]],
                           [bbox[2], bbox[1]],
                           [bbox[2], bbox[3]],
                           [bbox[0], bbox[3]]]
            refine_bboxes.append(refine_bbox)

        return refine_bboxes

    @staticmethod
    def get_out_img(img, bboxes, classes=None, scores=None, color=uImg.RED, thickness=2, alpha=0.):

        out_img = img.copy()
        for idx in range(len(bboxes)):
            pos = bboxes[idx]
            if all([True if x <= 2 else False for x in pos]):
                sz = img.shape[1::-1]
                pos = [int(pos[0] * sz[0]), int(pos[1] * sz[1]), int(pos[2] * sz[0]), int(pos[3] * sz[1])]
            text = classes[idx] if classes is not None else ''
            text += " : " + str(int(scores[idx] * 100)) if scores is not None else ''
            box_color = uImg.get_random_color(3) if isinstance(color, int) else color

            out_img = uImg.draw_box_on_img(out_img, pos, color=box_color, thickness=thickness, alpha=alpha)

            if text is not '':
                out_img = cv2.putText(out_img, text, (pos[0] + 4, pos[3] - 4), uImg.CV2_FONT, 0.5, uImg.BLACK, 6)
                out_img = cv2.putText(out_img, text, (pos[0] + 4, pos[3] - 4), uImg.CV2_FONT, 0.5, uImg.WHITE, 2)

                out_img = np.array(out_img)

        return out_img

    def make_person_boxed_image(self, img, results,
                                color=uImg.get_random_color(),
                                thickness=4,
                                imshow_sec=-1,
                                clockwise_=True):

        if isinstance(results, list):
            bboxes = results[0]
            labels = results[1]
            confs = results[2]
        else:
            bboxes = []
            labels = []
            confs = []

        draw_img = self.get_out_img(img.copy(), bboxes, labels, confs,
                                    color=color,
                                    thickness=thickness)

        uImg.imshow(draw_img, pause_sec=imshow_sec)

        return draw_img

    def process_person_detect(self, img):

        stt_time = time.time()

        results = self.detect_inst.run(img)

        return results, time.time() - stt_time

    def show_message(self, logger, msg, postfix="\n"):
        logger.info(msg)
        return msg + postfix

    def make_and_save_result_json(self, save_path, img_fname, img_shape, model_fname, method, results):
        result_json = dict()

        img_info = dict()
        img_info["image_name"] = os.path.basename(img_fname)
        img_info["attributes"] = {"color": img_shape[2],
                                  "image_width": img_shape[1],
                                  "image_height": img_shape[0],
                                  "image_path": img_fname}
        pers_info = dict()
        pers = dict()
        pers["algorithm"] = {"person_detection_algorithm": method,
                            "person_detection_model": os.path.splitext(os.path.basename(model_fname))[0]}
        pers["result"] = {"bboxes": results[0],
                         "labels": results[1],
                         "confidences": results[2]}
        pers_info["person"] = pers

        result_json["image_info"] = img_info
        result_json["person_info"] = pers_info

        with open(save_path, 'w', encoding='utf-8') as json_file:
            json.dump(result_json, json_file, indent="\t")


def main(args):

    this = PersDetect(ini=utils.get_ini_parameters(args.ini_fname))
    this.logger = utils.setup_logger_with_ini(this.ini['LOGGER'], logging_=args.logging_)
    this.logger.info(" # START {} in {} mode".format(_this_basename_, args.op_mode))
    this.init_out_dirs(args.out_path)
    this.init_detection_instance()

    if args.op_mode == 'image':

        img_fnames = sorted(utils.get_filenames(args.img_path,
                                                extensions=utils.IMG_EXTENSIONS,
                                                recursive_=True))
        this.logger.info(" # Get {:d} images to be processed in {}.".
                         format(len(img_fnames), args.img_path))

        msgs = ''
        tot_img_num = len(img_fnames)

        for idx, img_fname in enumerate(img_fnames):

            msg_prefix = " # {} : {:d}/{:d} #".format(this.method, idx+1, tot_img_num)
            base_fname = os.path.splitext(os.path.basename(img_fname))[0]

            img = uImg.imread(img_fname)

            results, proc_time = this.process_person_detect(img)

            msg = msg_prefix + " \"{}\" detected {:d} persons in {:4.2f} sec". \
                format(this.method, len(results[0]), proc_time)
            msgs += this.show_message(this.logger, msg)

            pers_img = this.make_person_boxed_image(img.copy(), results, color=uImg.GREEN)
            uImg.imwrite(pers_img, os.path.join(this.img_pers_dir,
                                                base_fname + IMAGE_FILE_EXT))

            this.make_and_save_result_json(os.path.join(this.json_pers_dir, base_fname + JSON_FILE_EXT),
                                           img_fname, img.shape, this.detect_inst.model_path, this.method, results)
    else:
        pass


def parse_arguments(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument("--op_mode", required=True, help="operation mode",
                        choices=['image'])
    parser.add_argument("--ini_fname", required=True, help="ini filename")

    parser.add_argument("--img_path", default=None, help="Input image path")
    parser.add_argument("--out_path", default='./Output/', help="Output path")

    parser.add_argument("--logging_", default=False, action='store_true', help="Logging flag")
    parser.add_argument("--console_logging_", default=False, action='store_true', help="Console logging flag")

    args = parser.parse_args(argv)

    args.img_path = utils.unicode_normalize(args.img_path)
    args.out_path = utils.unicode_normalize(args.out_path)

    return args


SELF_TEST_ = True
INI_FNAME = _this_basename_ + '.ini'

OP_MODE   = 'image'
IMG_PATH  = "./Input/"
OUT_PATH  = "./Output/Test/val2017_results/"


if __name__ == "__main__":

    if len(sys.argv) == 1:
        if SELF_TEST_:
            sys.argv.extend(["--op_mode", OP_MODE])
            sys.argv.extend(["--ini_fname", INI_FNAME])

            sys.argv.extend(["--img_path", IMG_PATH])
            sys.argv.extend(["--out_path", OUT_PATH])

            sys.argv.extend(["--logging_"])
            sys.argv.extend(["--console_logging_"])
        else:
            sys.argv.extend(["--help"])

    main(parse_arguments(sys.argv[1:]))
