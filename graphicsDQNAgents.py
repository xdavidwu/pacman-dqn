import random
import game
import graphicsUtils
from PIL import Image
from io import BytesIO
import re

import time

def getFrame():
        ps = graphicsUtils._canvas.postscript()
        ps_nofont = re.sub(r'^%%DocumentNeededResources: font [^\n]*\n', '', ps,
                count=1, flags=re.MULTILINE)
        ps_notext = re.sub(
                r'^/[^\n ]* findfont [^\n]* setfont\n[^\n]*\n[^\n]*\n[^\n]*\n[^\n]*DrawText\n',
                '', ps_nofont, count=1, flags=re.MULTILINE)
        return Image.open(BytesIO(ps_notext.encode('ascii')), formats=['EPS'])

class RandomSaveFrameAgent(game.Agent):
    def __init__(self):
        self.frame = 0

    def getAction(self, state):
        t = time.monotonic_ns();
        screen = getFrame()
        screen.load()
        #screen.save('pacman_%d.bmp' % self.frame)
        print(time.monotonic_ns() - t)
        self.frame += 1
        return random.choice(state.getLegalPacmanActions())
