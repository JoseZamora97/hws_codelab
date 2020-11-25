import argparse
import copy
import os
import queue
import random
from concurrent.futures.thread import ThreadPoolExecutor
from queue import Queue
from typing import List

from PIL import Image, ImageFilter
from tqdm import tqdm


class ImageOperations:
    config = {
        'radius_interval': [2, 10],
        'resize_interval': [0.25, 2.5]
    }

    @staticmethod
    def save(im: Image, path: str):
        """
        Saves the image
        :param im: the image
        :param path: root path file
        :return: None
        """
        im.save(path)

    @staticmethod
    def load(path) -> Image:
        """
        Loads an image from a path
        :param path: input path
        :return: an pillow.Image object
        """
        return Image.open(path)

    @staticmethod
    def blur(im: Image) -> Image:
        """
        Blurs an pillow.Image image
        :param im: the image
        :return: blurred pillow.Image object
        """
        return im.filter(ImageFilter.GaussianBlur(radius=random.uniform(
            *ImageOperations.config.get('radius_interval'))
        ))

    @staticmethod
    def resize(im: Image) -> Image:
        """
        Resizes an pillow.Image image
        :param im: the image
        :return: resized pillow.Image object
        """
        size_factor = random.uniform(*ImageOperations.config.get('resize_interval'))
        return im.resize((
            int(round(im.height * size_factor, 0)),
            int(round(im.width * size_factor, 0))
        ))

    @staticmethod
    def rotate(im: Image) -> Image:
        """
        Rotates an pillow.Image image
        :param im: the image
        :return: rotated pillow.Image object
        """
        return im.rotate(random.randint(0, 360))

    @staticmethod
    def transpose(im: Image) -> Image:
        """
        Transposes an pillow.Image image
        :param im: the image
        :return: transposed pillow.Image object
        """
        return im.transpose(
            random.choice([Image.FLIP_TOP_BOTTOM, Image.FLIP_LEFT_RIGHT])
        )


# Operations that will be randomly selected to apply.
operations = [
    ImageOperations.blur,
    ImageOperations.resize,
    ImageOperations.rotate,
    ImageOperations.transpose
]


def get_output_filename(item: str, root: str, i: int) -> str:
    """
    Auxiliary method, allows to create a path based on
    on path element (item) and the root path (root) by giving an index
    to the filepath. If i is negative it returns a path with the same
    name of the item but with different root.
    :param item: element path
    :param root: root path
    :param i: index
    :return: a path
    """
    element_split = item.split("/")
    item, ext = element_split[-1].split(".")
    if i < 0:
        return f"{root}/{'/'.join(element_split[:-1])}/{item}.{ext}"
    else:
        return f"{root}/{'/'.join(element_split[:-1])}/{item}_aug{i}.{ext}"


def augmentation(element: str, output: str, factor: int) -> None:
    """
    Applies a random modifications from the @operations list. And saves the
    modification on the output param.
    :param element: Element path of the image to be augmented
    :param output: Path folder where the augmentation will be saved
    :param factor: the augmentation factor.
    :return: None
    """

    out_filename = get_output_filename(element, output, -1)

    try:
        os.makedirs("/".join(out_filename.split("/")[:-1]))
    except:
        pass

    im = ImageOperations.load(element)
    ImageOperations.save(im, path=out_filename)

    for i in range(factor):
        out_filename = get_output_filename(element, output, i)
        im_aug = copy.deepcopy(im)
        for operation in set(random.sample(operations, k=random.randint(0, len(operations)))):
            im_aug = operation(im_aug)

        ImageOperations.save(im_aug, path=out_filename)


def get_images_paths(path: str) -> List[str]:
    """
    Iterates over the path and search for images/<image.jpeg> files
    and saves in a list.
    :param path: the path to be iterated
    :return: the list of image paths.
    """

    image_paths = []

    for folder in os.listdir(path):
        for file in os.listdir(os.path.join(f"{path}/{folder}", "images")):
            image_paths.append(f"{path}/{folder}/images/{file}")

    return image_paths


def execute_augmentation(queue_images: Queue, progress: tqdm, output: str, factor: int) -> None:
    """
    Worker action to be called from the ThreadPoolExecutor.
    :param queue_images: the queue of work.
    :param progress: the progress object to be updated
    :param output: the output path to save the results.
    :param factor: the factor of augmentation
    :return: None
    """
    while not queue_images.empty():
        element = queue_images.get(block=False)
        augmentation(element, output, factor)
        progress.update(1)


def action(args):
    # Check the factor
    assert args.factor is not None, "Augmentation Factor can't be None"
    assert args.factor >= 1, "Augmentation Factor can't be less than 1"
    # Check the workers
    assert args.workers is not None, "Workers can't be None"
    assert args.workers >= 1, "Workers can't be less than 1"
    # Check the Input path
    assert os.path.exists(args.input), "Input path doesn't exist"
    assert os.path.isdir(args.input), "Input path must be a folder"
    # Check the Output path
    assert not os.path.exists(args.output), "Output path already exist"

    # Take the arguments params.
    augmentation_factor: int = args.factor

    path_input: str = args.input
    path_output: str = args.output

    # Set the max workers
    max_workers = args.workers * 5

    # Get the list of images
    images = get_images_paths(path=path_input)

    # Create the queue of jobs and the progress object
    queue_images = Queue()
    queue_images.queue = queue.deque(images)
    progress = tqdm(total=queue_images.qsize(), position=0, desc="Executing...")

    # Create the ThreadPoolExecutor and and submit the jobs.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(max_workers):
            executor.submit(execute_augmentation,
                            queue_images,
                            progress,
                            path_output,
                            augmentation_factor)

    # Shutdown the executor waiting till the jobs are done
    executor.shutdown(wait=True)


def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description="Data Augmentation")
    # Add parser input.
    parser.add_argument("-i", "--input_dataset", required=True,
                        help="Set the Imagenet input directory")
    # Add parser output.
    parser.add_argument("-o", "--output", required=True,
                        help="Set the Imagenet augmented root directory")
    # Add parser factor.
    parser.add_argument("-f", "--factor", type=int, required=True,
                        help="Set the factor of augmentation")
    # Add parser workers.
    parser.add_argument("-w", "--workers", default=None, type=int,
                        help="Set the amount of threads you want to use")

    # Set the parser function.
    parser.set_defaults(func=action)
    # Parse the arguments.
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    # --input_dataset =./tiny_imagenet --factor = 20 --output_dataset =./augmented_tiny_imagenet
    parse_arguments()
