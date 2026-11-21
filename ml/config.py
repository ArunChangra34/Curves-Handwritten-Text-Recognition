"""Shared constants for Curves."""
IMG_H, IMG_W = 32, 160          # model input size (height, width)
MAX_LEN = 16                    # max characters per sample
SEED = 7
TRAIN_FONTS = {
    "en": ["Caveat.ttf", "IndieFlower-Regular.ttf", "Kalam-Regular.ttf", "Kalam-Light.ttf"],
    "hi": ["Kalam-Regular.ttf", "Kalam-Light.ttf", "Kalam-Bold.ttf", "Amita-Regular.ttf",
           "Sahitya-Regular.ttf", "Karma-Regular.ttf"],
}
# fonts never seen in training -> honest generalisation test
HELD_FONTS = {"en": ["PatrickHand-Regular.ttf"], "hi": ["Sumana-Regular.ttf", "Laila-Regular.ttf"]}
