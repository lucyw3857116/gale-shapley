import sys
import numpy as np 

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Incorrect Usage: python validate.py <file1> <file2>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    with open(input_file, "r") as f:
        header = f.readline().strip()
        num_part, _, _, num_pref = header.split()
        num_participants = int(num_part)
        num_preferences = int(num_pref)

        pref_list = []
        for line in f:
            prefs = line.strip().split()
            pref_list.append(prefs)

    with open(output_file, "r") as f:
        output_data = []
        for line in f:
            output_data.append(line.strip().split())

    matches = dict()
    for line in output_data:
        matches[int(line[0])] = int(line[1])
        matches[int(line[1])] = int(line[0])

    # for each pair check all other possible matches and see if they are ranked higher
    stable = True
    for m in range(0,num_participants//2):
        for f in range(num_participants//2, num_participants):
            male_id = str(m)
            female_id = str(f)
            male_match = matches[m]
            female_match = matches[f]

            # check if this pair ranks each other higher than the chosen pair
            if female_id in pref_list[m] and male_id in pref_list[f]:
                check1 = pref_list[m].index(female_id) < pref_list[m].index(str(male_match))
                check2 = pref_list[f].index(male_id) < pref_list[f].index(str(female_match))

                if (check1 and check2):
                    print(f"Blocking pair found: (%d,%d) over (%d, %d) and (%d, %d)" % (m, f, m, male_match, f, female_match))
                    stable = False
                    break

        if not stable:
            break
    
    if stable:
        print("No blocking pair found, correct!")

