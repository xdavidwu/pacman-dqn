import random
import game
import graphicsUtils
from PIL import Image
from io import BytesIO

class RandomSaveFrameAgent(game.Agent):
    def __init__(self):
        self.frame = 0

    def getAction(self, state):
        global _canvas
        ps = graphicsUtils._canvas.postscript()
        screen = Image.open(BytesIO(ps.encode('utf8')), formats=['EPS'])
        screen.load()
        self.frame += 1
        return random.choice(state.getLegalPacmanActions())
