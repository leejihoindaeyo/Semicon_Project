import os
import cv2
import torch
import numpy as np
import time
from hoonpyutils import UtilsCommon as utils

from .image import get_affine_transform
from .post_process import ctdet_post_process
from .model_res import create_model, load_model
from .decode import ctdet_decode
import torch.backends.cudnn as cudnn
from .utils import load_class_names


class CenterNet:
    def __init__(self, gpu_list='1', ini=None, logger=utils.get_stdout_logger()):
        self.model_path = None
        self.label_path = None
        self.torch_num_threads = 4
        self.gpu_list = gpu_list
        self.arch = 'res_18'
        self.device = None
        self.detector = None

        self.num_classes = None
        self.class_names = None
        self.heads = {'hm': 1, 'wh': 2, 'reg': 2}
        self.head_conv = 64
        self.mean = [0.408, 0.447, 0.47]
        self.std = [0.289, 0.274, 0.278]
        self.max_per_image = 100
        self.scales = [1.0]
        self.reg_offset = True
        self.pause = True
        self.pad = 31

        self.input_h = 512
        self.input_w = 512

        self.down_ratio = 4
        self.fix_res = True
        # self.nms = False
        self.vis_thresh = 0.3
        self.flip_test = True

        self.logger = logger

        if ini is not None:
            self.init_ini(ini)
            self.init_net()

    def init_ini(self, ini):
        self.model_path = ini['model_path'].strip()
        self.label_path = ini['label_path'].strip()
        self.class_names = load_class_names(self.label_path)
        self.num_classes = len(self.class_names)
        # self.heads = {'hm': self.num_classes, 'wh': 4}
        if 'torch_num_threads' in ini:
            self.torch_num_threads = int(ini['torch_num_threads'])

        self.logger.info(" # Loading pre-trained CenterNet person detector...")

    def init_net(self):
        torch.set_num_threads(self.torch_num_threads)
        self.logger.info(" # torch_num_thread : {}".format(self.torch_num_threads))

        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_list

        self.logger.info(" # CenterNet : loading pre-trained model, {}.".format(self.model_path))

        self.device = torch.device('cuda')

        self.logger.info('Creating model...')
        self.model = create_model(self.arch, self.heads, self.head_conv)
        self.logger.info('Loading model...')
        self.model = load_model(self.model, self.model_path)
        self.model = self.model.to(self.device)
        # self.model = self.model.cuda()
        # self.model = torch.nn.DataParallel(self.model)
        cudnn.benchmark = False

        self.model.eval()

        self.mean = np.array(self.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(self.std, dtype=np.float32).reshape(1, 1, 3)

        self.logger.info(" # CenterNet : Loaded.")

    def run(self, image, meta=None):
        pre_time, net_time, dec_time, post_time = 0, 0, 0, 0
        merge_time, tot_time = 0, 0
        bbox_arr = []
        class_arr = []
        conf_arr = []
        start_time = time.time()
        detections = []
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # height, width = image.shape[0:2]
        # image = cv2.resize(image, (self.input_w, self.input_h), interpolation=cv2.INTER_CUBIC)
        # rollback_factor = [width / self.input_w, height / self.input_h]

        time_list = []
        for scale in self.scales:
            scale_start_time = time.time()
            time_list.append(scale_start_time)

            images, meta = self.pre_process(image, scale, meta)
            time_list.append(time.time())

            images = images.to(self.device)
            torch.cuda.synchronize()
            time_list.append(time.time())

            pre_process_time = time.time()
            pre_time += pre_process_time - scale_start_time

            output, dets, forward_time = self.process(images, return_time=True)
            torch.cuda.synchronize()
            time_list.append(time.time())

            net_time += forward_time - pre_process_time
            decode_time = time.time()
            dec_time += decode_time - forward_time

            dets = self.post_process(dets, meta, scale)
            torch.cuda.synchronize()
            time_list.append(time.time())

            post_process_time = time.time()
            post_time += post_process_time - decode_time

            detections.append(dets)

        results = self.merge_outputs(detections)
        torch.cuda.synchronize()
        end_time = time.time()
        merge_time += end_time - post_process_time
        tot_time += end_time - start_time

        for i in range(1, self.num_classes + 1):
            for bbox in results[i]:
                if bbox[4] > self.vis_thresh:

                    rel_bbox = self.calculate_bbox_relative_position(image.shape, bbox[:4])
                    bbox_arr.append(rel_bbox)
                    class_arr.append(self.class_names[i-1])  # only 1 class
                    conf_arr.append(float(bbox[4]))

        detected_pers = [bbox_arr, class_arr, conf_arr]

        elapse_list = []
        for idx in range(1, len(time_list)):
            elapse_list.append("{:.4f}".format(time_list[idx] - time_list[idx - 1]))

        self.logger.info(" # pre {}".format(elapse_list))

        self.logger.info(" # tot: {:.3f}s | pre: {:.3f}s | net: {:.3f}s | "
                         "dec: {:.3f}s | post: {:.3f}s | merge: {:.3f}s | "
                         .format(tot_time, pre_time, net_time, dec_time, post_time, merge_time))

        # torch.cuda.empty_cache()

        return detected_pers

    def pre_process(self, image, scale, meta=None):
        height, width = image.shape[0:2]
        new_height = int(height * scale)
        new_width = int(width * scale)
        if self.fix_res:
            inp_height, inp_width = self.input_h, self.input_w
            c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
            s = max(height, width) * 1.0
        else:
            inp_height = (new_height | self.pad) + 1
            inp_width = (new_width | self.pad) + 1
            c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
            s = np.array([inp_width, inp_height], dtype=np.float32)

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(image, (new_width, new_height))
        inp_image = cv2.warpAffine(
            resized_image, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
        images = torch.from_numpy(images)
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.down_ratio,
                'out_width': inp_width // self.down_ratio}
        return images, meta

    def process(self, images, return_time=False):
        with torch.no_grad():
            output = self.model(images)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg'] if self.reg_offset else None
            # reg = wh[:, 2:, :, :]
            # wh = wh[:, :2, :, :]

            torch.cuda.synchronize()
            forward_time = time.time()
            dets = ctdet_decode(hm, wh, reg=reg)

        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.num_classes)
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)
            # if len(self.scales) > 1 or self.opt.nms:
            #     soft_nms(results[j], Nt=0.5, method=2)
        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    @staticmethod
    def calculate_bbox_relative_position(img_size, bbox):
        h = img_size[0]
        w = img_size[1]

        rel_bbox = [float(bbox[0] / w), float(bbox[1] / h), float(bbox[2] / w), float(bbox[3] / h)]

        return rel_bbox