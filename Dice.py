import random

class Dice:
    def roll(self):
        x=random.randint(1,6)
        y=random.randint(1,6)
        return x,y

dice=Dice()
dice2=Dice()
print(dice.roll())
print(dice2.roll() )
   
