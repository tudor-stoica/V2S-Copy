import subprocess

# List of values for i
i_values = [25, 50, 100, 150, 160]
# Loop through each value and run the command
for i in i_values:
    command = f"python v2s_main.py --seg {i}"
    print(f"Running command: {command}")
    subprocess.run(command, shell=True)
