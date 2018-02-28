import os
import ntpath
import time
import util
import html


class Visualizer():
    def __init__(self, name, checkpoints_dir, win_size=256):
        self.win_size = win_size
        self.name = name

        self.web_dir = os.path.join(checkpoints_dir, 'web')
        self.img_dir = os.path.join(self.web_dir, 'images')
        print('create web directory %s...' % self.web_dir)
        util.mkdirs([self.web_dir, self.img_dir])

        self.log_name = os.path.join(checkpoints_dir, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write(
                '================ Training Loss (%s) ================\n' % now
            )

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch):
        for label, image_numpy in visuals.items():
            img_path = os.path.join(
                self.img_dir, 'epoch%.3d_%s.png' % (epoch, label)
            )
            util.save_image(image_numpy, img_path)
        # update website
        webpage = html.HTML(
            self.web_dir, 'Experiment name = %s' % self.name, reflesh=1
        )
        for n in range(epoch, 0, -1):
            webpage.add_header('epoch [%d]' % n)
            ims = []
            txts = []
            links = []

            for label, image_numpy in visuals.items():
                img_path = 'epoch%.3d_%s.png' % (n, label)
                ims.append(img_path)
                txts.append(label)
                links.append(img_path)
            webpage.add_images(ims, txts, links, width=self.win_size)
        webpage.save()

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        name = ntpath.basename(image_path)

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
