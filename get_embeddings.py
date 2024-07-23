import pandas as pd
from img2vec_pytorch import Img2Vec
import image_utils

try:
    paths = image_utils.get_images_from_dir("processed_images/horse")
    images = [image_utils.load_image(path) for path in paths]

    # Initialize Img2Vec with GPU
    img2vec = Img2Vec(cuda=True)
    embeddings = img2vec.get_vec(images)

    print(embeddings.shape)

    df = pd.DataFrame(embeddings)
    df["filepaths"] = paths
    df.to_csv("embeddings/horse_embeddings.csv", index=False)
except ImportError as e:
    print(f"An import error occurred: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
