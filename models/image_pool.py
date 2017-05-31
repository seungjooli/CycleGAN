import numpy as np


class ImagePool:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images

        output_images = []
        for image in images:
            if len(self.images) < self.pool_size:
                self.images.append(image)
                output_images.append(image)
            else:
                p = np.random.rand()
                if p > 0.5:
                    random_id = np.random.randint(0, self.pool_size)
                    output_images.append(self.images[random_id])
                    self.images[random_id] = image
                else:
                    output_images.append(image)

        return output_images
