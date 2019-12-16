def calculatePrice(foodClassified):
    # {
    #     fried_chicken:{
    #         price :
    #         total_price :
    #         qty :
    #     }
    # }
    totalPrice = 0
    newDict = {}
    for foodItem,quantity in foodClassified.items() :
        if foodItem == "fried_chicken" :
            newDict.update({
                "fried_chicken" : {
                    "price" : "3.00",
                    "qty" : quantity
                }})
            totalPrice += float(quantity) * float(newDict["fried_chicken"]["price"])
        elif foodItem == "rice" :
            newDict.update({
                "rice" : {
                    "price" : "1.00",
                    "qty" : quantity
                }})
            totalPrice += float(quantity) * float(newDict["rice"]["price"])
        elif foodItem == "mixed_vegetables" :
            newDict.update({
                "mixed_vegetables" : {
                    "price" : "1.50",
                    "qty" : quantity
                }})
            totalPrice += float(quantity) * float(newDict["mixed_vegetables"]["price"])
        elif foodItem == "fried_egg" :
            newDict.update({
                "fried_egg" : {
                    "price" : "1.00",
                    "qty" : quantity
                }})
            totalPrice += float(quantity) * float(newDict["fried_egg"]["price"])
    newDict.update({"grand_total" : "{:.2f}".format(totalPrice)})
    return newDict