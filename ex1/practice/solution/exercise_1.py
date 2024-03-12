# Script that checks whether a year is a leap year or not.

if __name__ == "__main__":
    # read in string of years
    years_str = input("Enter one or more years: ")

    # convert string of years into list of numbers
    years = [int(y) for y in years_str.split()]

    # check if year is a leap year
    for year in years:
        if year < 0:
            print(year, "is not a valid input.")
            continue
        if year % 4 == 0:
            if year % 100 == 0:
                if year % 400 == 0:
                    print("The year", year, "is a leap year.")
                else:
                    print("The year", year, "is not a leap year.")
            else:
                print("The year", year, "is a leap year.")
        else:
            print("The year", year, "is not a leap year.")
