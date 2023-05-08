import sys



def main():

    names = sys.argv[1]

    ages = {}
    genders = {0: 0, 1: 0}
    ethnicities = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    # Open the file for reading
    with open(names, 'r') as f:
        # Read the contents of the file into a list of strings
        filenames = f.read().splitlines()

        for filename in filenames:
            parts = filename.split("_") 
            age = int(parts[0])
            if age in ages:
                ages[age] += 1
            else:
                ages[age] = 1
            gender = int(parts[1])
            genders[gender] += 1
            ethnicity = int(parts[2])
            ethnicities[ethnicity] += 1

    total = len(filenames)

    print(f"total {total}")

    
    print("ages")
    print(ages)
    print("genders")
    print(genders)
    print("ethnicities")
    print(ethnicities)

    print(f"Male: {genders[0]/total * 100} and Female: {genders[1]/total * 100}")
    over_80 = 0
    under_5 = 0
    for age in ages:
        if age >= 80:
            over_80 += ages[age]
        if age <= 5:
            under_5 += ages[age]
    print(f"5 and under: {(under_5/total) * 100}")
    print(f"80 and over: {(over_80/total) * 100}")
    print(f"White: {ethnicities[0]/total * 100}, Black: {ethnicities[1]/total * 100}")
    print(f"Asian: {ethnicities[2]/total * 100}, Indian: {ethnicities[3]/total * 100}, Other: {ethnicities[4]/total * 100}")
# -----------------------------------
if __name__ == '__main__':
    main()