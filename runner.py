import subprocess

i_values = [1, 3, 5, 10, 15]
place = ["start", "center", "end"]
stack = [True, False]

# Loop through each value and run the command
for s in stack:
    for p in place:
        for i in i_values:
            if s:
                command = f"python v2s_main.py --seg {i} --place {p} --stack"
            else:
                command = f"python v2s_main.py --seg {i} --place {p}"
            print(f"Running command: {command}")
            subprocess.run(command, shell=True)

        # After each `place` loop, add two new lines in `out.txt` for separation
        with open("out.txt", "a") as file:
            file.write("\n\n")  # Add two new lines

    # After each `stack` loop, add four new lines in `out.txt` for further separation
    with open("out.txt", "a") as file:
        file.write("\n\n\n\n")  # Add four new lines
