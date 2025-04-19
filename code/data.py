import random
import math

def generate_gale_shapley_input(n, filename="input.txt"):
    with open(filename, "w") as f:
        f.write(f"{n}\n")  # First line: just n

        # Generate men's preference lists
        sqrt_n = n
        for _ in range(n):
            prefs = list(range(1, sqrt_n + 1))
            random.shuffle(prefs)
            f.write(f"{sqrt_n} " + " ".join(map(str, prefs)) + "\n")

        # Generate women's preference lists
        for _ in range(n):
            prefs = list(range(1, sqrt_n + 1))
            random.shuffle(prefs)
            f.write(f"{sqrt_n} " + " ".join(map(str, prefs)) + "\n")

# Example usage
# generate_gale_shapley_input(1000, "1000_1000.txt")
# print("1000_1000.txt generated")
# generate_gale_shapley_input(2000, "2000_2000.txt")
# print("2000_2000.txt generated")
generate_gale_shapley_input(4000, "4000_4000.txt")
print("4000_4000.txt generated")
# generate_gale_shapley_input(6000, "6000_6000.txt")
# print("6000_6000.txt generated")
# generate_gale_shapley_input(10000, "10000_10000.txt")
# print("10000_10000.txt generated")



# largest data is probably 6000