import sys
import numpy as np 

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Incorrect Usage: python validate.py <file1> <file2>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    input_data = np.genfromtxt(input_file, delimiter="\t", dtype=None, encoding=None)    
    num_part, _, _, num_pref = input_data[0].split(" ")
    num_participants = int(num_part)
    num_preferences = int(num_pref)
    input_data = input_data[1:]

    pref_list = list()
    for line in input_data:
        pref_list.append(line.split(" "))
    
    output_data = np.genfromtxt(output_file, delimiter="\t", dtype=None, encoding=None)
    matches = dict()
    for line in output_data:
        line = line.split(" ")
        matches[line[0]] = line[1]
        matches[line[1]] = line[0]
    # for each pair check all other possible matches and see if they are ranked higher
    stable = True
    for m in range(0,num_participants//2):
        for f in range(num_participants//2, num_participants):
            male_id = str(m)
            female_id = str(f)
            male_match = matches[male_id]
            female_match = matches[female_id]

            # check if this pair ranks each other higher than the chosen pair
            if female_id not in pref_list[m]:
                check1 = True
            else:
                check1 = pref_list[m].index(female_id) < pref_list[m].index(male_match)
            
            if male_id not in pref_list[f]:
                check2 = True
            else:
                check2 = pref_list[f].index(male_id) < pref_list[f].index(female_match)

            if (check1 and check2):
                print(f"Blocking pair found: {male_id} and {female_id}, incorrect")
                stable = False
                break

        if not stable:
            break
    
    if stable:
        print("No blocking pair found, correct!")

