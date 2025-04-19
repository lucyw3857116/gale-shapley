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
        num_participants = int(header)*2
        pref_list = []
        count = 0
        for line in f:
            prefs = line.strip().split()
            if count < num_participants//2:
                prefs = [int(p) - 1 + num_participants//2 for p in prefs[1:]]
            else:
                prefs = [int(p) - 1 for p in prefs[1:]]
            count += 1
            pref_list.append(prefs)

    with open(output_file, "r") as f:
        output_data = []
        for line in f:
            output_data.append(line.strip().split())

    matches = dict()
    for line in output_data:
        m = int(line[0])
        w = int(line[1])
        if w in matches.keys():
            print("Two men with same woman ("+line[1]+"): "+line[0]+" and ", str(matches[w]))
        matches[m] = w
        matches[w] = m
    # print(matches)

    # for each pair check all other possible matches and see if they are ranked higher
    stable = True
    mCount = 0
    fCount = 0
    for m in range(0,num_participants//2):
        mCount += 1
        for f in range(num_participants//2, num_participants):
            fCount += 1
            male_id = str(m)
            female_id = str(f)
            if m not in matches.keys():
                print("There was no match for m = " + str(m) + " \n")
                stable = False
                break
            if f not in matches.keys():
                print("There was no match for f = " + str(f) + " \n")
                stable = False
                break
            male_match = matches[m]
            # print(f)
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
    print(mCount, fCount)

