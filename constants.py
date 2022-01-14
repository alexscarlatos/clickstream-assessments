from enum import Enum

ASSISTIVE_EVENT_IDS = [4, 8, 17, 24, 27, 34] # Open Calculator, Eliminate Choice, Scratchwork Mode On, Highlight, TextToSpeech, Increase Zoom

class Correctness(Enum):
    INCOMPLETE = 0
    INCORRECT = 1
    CORRECT = 2
    WORKING = 3
