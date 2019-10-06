# {'fried_chicken': '2'}
def calculatePrice(foodClassified):
    total_price = 0
    for foodItem,quantity in foodClassified.items() :
        if foodItem == "fried_chicken":
            total_price += int(quantity)*2
        elif foodItem == "rice":
            total_price += int(quantity)*1
        elif foodItem == "mixed_vegetables":
            total_price += int(quantity)*1.5
        elif foodItem == "fried_egg":
            total_price +=int(quantity)*1
    return total_price